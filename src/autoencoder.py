from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .atributos import FEATURES
from .comum import escrever_json, ler_json, parquet, sha256
from .dados import lotes


class Autoencoder(nn.Module):
    def __init__(self, input_dim=21, hidden_dims=(16, 8), latent_dim=4):
        super().__init__()
        first, second = hidden_dims
        layers = []
        dims = [input_dim, first, second, latent_dim, second, first, input_dim]
        for index, (left, right) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(dims)-2:
                layers.append(nn.LeakyReLU(.1))
        self.network = nn.Sequential(*layers)

    def forward(self, values):
        return self.network(values)


def configurar(seed, requested):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA solicitada e indisponível; não há troca automática para CPU.')
    if requested not in ('cpu', 'cuda'):
        raise ValueError('Dispositivo deve ser cpu ou cuda.')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.device(requested)


def salvar_torch(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    torch.save(data, temp)
    temp.replace(path)


def preparar_memmap(con, files, folder, contract, batch_size, modulus):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / 'matrizes.json'
    targets = [folder/'treino.npy', folder/'validacao_interna.npy']
    if marker.exists():
        record = ler_json(marker)
        if record['contract'] != contract:
            raise ValueError('Matrizes de treino pertencem a outro contrato.')
        if all(p.exists() and sha256(p) == record['hashes'][p.name] for p in targets):
            return tuple(np.load(p, mmap_mode='r') for p in targets), record
        raise ValueError('Matriz mapeada alterada ou ausente; use outra execução.')
    total, val_rows = con.execute(f'SELECT count(*),sum((row_id % {modulus}=0)::BIGINT) FROM {parquet(files)}').fetchone()
    fit_rows = total-val_rows
    if min(fit_rows, val_rows) < 1:
        raise ValueError('Treino ou validação interna vazio.')
    temps = [p.with_name(p.name+'.partial') for p in targets]
    arrays = [np.lib.format.open_memmap(p, mode='w+', dtype=np.float32, shape=(n, len(FEATURES))) for p, n in zip(temps, (fit_rows,val_rows))]
    offsets = [0, 0]
    for ids, values in lotes(files, batch_size):
        mask = ids % modulus == 0
        for k, selected in enumerate((values[~mask], values[mask])):
            arrays[k][offsets[k]:offsets[k]+len(selected)] = selected
            offsets[k] += len(selected)
    if offsets != [fit_rows, val_rows]:
        raise ValueError('Cobertura de treino alterada durante a leitura.')
    for arr in arrays:
        arr.flush()
        arr._mmap.close()
    del arrays
    for temp, target in zip(temps, targets):
        temp.replace(target)
    record = dict(contract=contract, rows_total=total, rows_fit=fit_rows, rows_internal_validation=val_rows,
                  hashes={p.name:sha256(p) for p in targets}, shuffle='permutacao_global_numpy_seed_mais_epoca_em_CPU')
    escrever_json(marker, record)
    return tuple(np.load(p, mmap_mode='r') for p in targets), record


def treinar(con, files, settings, seed, folder, contract, scratch):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    device = configurar(seed, settings['device'])
    (train, validation), rows = preparar_memmap(con, files, scratch, contract, settings['batch_size'], settings['internal_validation_modulus'])
    model = Autoencoder(hidden_dims=tuple(settings['hidden_dims']), latent_dim=settings['latent_dim']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    use_amp = settings['amp'] and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    criterion = nn.MSELoss()
    checkpoint = folder/'checkpoint.pt'
    first_epoch, history, best_loss, best_state, best_epoch, stale = 0, [], float('inf'), None, 0, 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['contract'] != contract:
            raise ValueError('Checkpoint de outro experimento.')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['amp_scaler'])
        first_epoch, history = state['epoch'], state['history']
        best_loss, best_state, best_epoch, stale = (state[k] for k in ('best_loss', 'best_state', 'best_epoch', 'stale'))
        torch.set_rng_state(state['rng_cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['rng_cuda'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for epoch in range(first_epoch, settings['max_epochs']):
            if stale >= settings['patience']:
                break
            model.train()
            order = np.random.default_rng(seed+epoch).permutation(len(train))
            train_sum = 0.
            for offset in range(0, len(order), settings['batch_size']):
                values = np.array(train[order[offset:offset+settings['batch_size']]], copy=True)
                batch = torch.from_numpy(values).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    loss = criterion(model(batch), batch)
                if not torch.isfinite(loss):
                    raise ValueError('Perda AE não finita; treino interrompido.')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_sum += float(loss.detach())*len(values)
            del order
            model.eval()
            val_sum = 0.
            with torch.inference_mode():
                for offset in range(0, len(validation), settings['batch_size']*2):
                    batch = torch.from_numpy(np.array(validation[offset:offset+settings['batch_size']*2], copy=True)).to(device)
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        loss = criterion(model(batch), batch)
                    if not torch.isfinite(loss):
                        raise ValueError('Perda interna de validação não finita.')
                    val_sum += float(loss)*len(batch)
            val_loss = val_sum/len(validation)
            improved = val_loss < best_loss-settings['min_delta']
            if improved:
                best_loss, best_epoch, stale = val_loss, epoch+1, 0
                best_state = copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
            else:
                stale += 1
            history.append(dict(epoch=epoch+1, train_loss=train_sum/len(train), internal_validation_loss=val_loss, improved=improved))
            salvar_torch(checkpoint, dict(contract=contract, epoch=epoch+1, model=model.state_dict(), optimizer=optimizer.state_dict(), amp_scaler=scaler.state_dict(), best_loss=best_loss, best_state=best_state, best_epoch=best_epoch, stale=stale, history=history, rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else []))
            escrever_json(folder/'historico.json', history)
            print(f'AE época {epoch+1}: treino={train_sum/len(train):.7g}; validação interna={val_loss:.7g}', flush=True)
        if best_state is None:
            raise ValueError('AE não produziu um estado válido.')
        salvar_torch(folder/'modelo.pt', dict(state_dict=best_state, features=FEATURES, settings=settings, contract=contract, best_epoch=best_epoch))
        escrever_json(folder/'historico.json', history)
        return dict(**rows, best_epoch=best_epoch, epochs=len(history), best_internal_validation_loss=best_loss,
                    gpu_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
    finally:
        train._mmap.close()
        validation._mmap.close()


def carregar(path, device_name):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['features'] != FEATURES:
        raise ValueError('Contrato B incompatível no modelo AE.')
    device = torch.device(device_name)
    model = Autoencoder(hidden_dims=tuple(state['settings']['hidden_dims']), latent_dim=state['settings']['latent_dim']).to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    def score(values):
        with torch.inference_mode():
            batch = torch.from_numpy(values).to(device)
            return torch.mean((model(batch).float()-batch.float())**2, dim=1).cpu().numpy()
    return score
