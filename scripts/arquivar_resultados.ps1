param(
    [ValidateSet('Copiar','RemoverOrigem')]
    [string]$Acao = 'Copiar',
    [string]$Destino = 'I:\arquivo-sbrc-2027\anual-2025-b-03'
)

$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$expectedSource = Join-Path $projectPath 'resultados\anual-2025-b-03'
$sourcePath = (Resolve-Path -LiteralPath $expectedSource).Path
if ($sourcePath -ne $expectedSource) { throw 'Origem fora da execução esperada.' }
$destinationPath = [IO.Path]::GetFullPath($Destino).TrimEnd('\')
if ([IO.Path]::GetPathRoot($destinationPath) -eq [IO.Path]::GetPathRoot($sourcePath)) {
    throw 'O destino deve estar em outra unidade.'
}
if ($destinationPath.StartsWith($sourcePath+'\',[StringComparison]::OrdinalIgnoreCase) -or
    $sourcePath.StartsWith($destinationPath+'\',[StringComparison]::OrdinalIgnoreCase)) {
    throw 'Origem e destino devem ser separados.'
}
function Assert-Idle {
    foreach ($relative in @('.lock','operacao\.supervisor.lock')) {
        if (Test-Path -LiteralPath (Join-Path $sourcePath $relative)) {
            throw 'Há um bloqueio na execução; encerre o processamento antes de arquivar.'
        }
    }
    $manifest = Get-Content -LiteralPath (Join-Path $sourcePath 'manifesto.json') -Raw | ConvertFrom-Json
    if ($manifest.status -ne 'completed') { throw 'A execução precisa estar concluída.' }
}
function Get-Files([string]$Folder) {
    $items = @(Get-ChildItem -LiteralPath $Folder -Force -Recurse)
    $rootItem = Get-Item -LiteralPath $Folder -Force
    if (($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
        @($items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
        throw 'Links ou junctions não são admitidos no arquivamento.'
    }
    return @($items | Where-Object { -not $_.PSIsContainer } | Sort-Object FullName)
}
Assert-Idle
$sourceFiles = @(Get-Files $sourcePath)
if (-not $sourceFiles.Count) { throw 'Origem vazia.' }
$parentPath = Split-Path -Parent $destinationPath
$statePath = Join-Path $parentPath ((Split-Path -Leaf $destinationPath)+'-arquivamento.json')

if ($Acao -eq 'Copiar') {
    if ((Test-Path -LiteralPath $destinationPath) -and -not (Test-Path -LiteralPath $statePath)) {
        throw 'Destino existente sem registro de arquivamento; escolha uma pasta nova.'
    }
    if (Test-Path -LiteralPath $statePath) {
        $prior = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        if ($prior.source -ne $sourcePath -or $prior.destination -ne $destinationPath) {
            throw 'O registro pertence a outra transferência.'
        }
    }
    New-Item -ItemType Directory -Path $parentPath -Force | Out-Null
    [ordered]@{ source=$sourcePath; destination=$destinationPath; status='copying' } |
        ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
    $logPath = Join-Path $parentPath ((Split-Path -Leaf $destinationPath)+'-copia.log')
    & robocopy $sourcePath $destinationPath /E /Z /COPY:DAT /DCOPY:DAT /R:1 /W:1 /XJ /NP "/LOG+:$logPath"
    if ($LASTEXITCODE -ge 8) { throw "Falha na cópia. Consulte $logPath. A origem foi preservada." }
} else {
    if (-not (Test-Path -LiteralPath $statePath)) { throw 'Nenhum registro de cópia encontrado.' }
    $prior = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ($prior.status -ne 'verified' -or $prior.source -ne $sourcePath -or $prior.destination -ne $destinationPath) {
        throw 'A cópia precisa estar conferida antes da remoção da origem.'
    }
}

Assert-Idle
$sourceFiles = @(Get-Files $sourcePath)
$destinationFiles = @(Get-Files $destinationPath)
if ($sourceFiles.Count -ne $destinationFiles.Count) { throw 'Quantidade de arquivos divergente; origem preservada.' }
$checks = @()
for ($i=0; $i -lt $sourceFiles.Count; $i++) {
    $file = $sourceFiles[$i]
    $relative = $file.FullName.Substring($sourcePath.Length+1)
    $copy = Join-Path $destinationPath $relative
    if (-not (Test-Path -LiteralPath $copy -PathType Leaf) -or
        (Get-Item -LiteralPath $copy).Length -ne $file.Length) {
        throw "Arquivo ausente ou tamanho divergente: $relative. Origem preservada."
    }
    $digest = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    if ($digest -ne (Get-FileHash -LiteralPath $copy -Algorithm SHA256).Hash) {
        throw "Hash divergente: $relative. Origem preservada."
    }
    $checks += [ordered]@{ path=$relative; bytes=$file.Length; sha256=$digest }
    if (($i+1)%100 -eq 0 -or $i+1 -eq $sourceFiles.Count) {
        Write-Host "Arquivos conferidos: $($i+1)/$($sourceFiles.Count)"
    }
}
Assert-Idle
$finalSource = @(Get-Files $sourcePath)
$finalDestination = @(Get-Files $destinationPath)
if ($finalSource.Count -ne $sourceFiles.Count -or $finalDestination.Count -ne $destinationFiles.Count) {
    throw 'A quantidade de arquivos mudou durante a conferência.'
}
for ($i=0; $i -lt $sourceFiles.Count; $i++) {
    foreach ($pair in @(@($sourceFiles[$i],$finalSource[$i]),@($destinationFiles[$i],$finalDestination[$i]))) {
        if ($pair[0].FullName -ne $pair[1].FullName -or $pair[0].Length -ne $pair[1].Length -or
            $pair[0].LastWriteTimeUtc.Ticks -ne $pair[1].LastWriteTimeUtc.Ticks) {
            throw 'Um arquivo mudou durante a conferência; origem preservada.'
        }
    }
}
if ($Acao -eq 'RemoverOrigem') {
    $resolvedSource = (Resolve-Path -LiteralPath $expectedSource).Path
    if ($resolvedSource -ne $sourcePath -or $sourcePath -ne (Join-Path $projectPath 'resultados\anual-2025-b-03')) {
        throw 'Destino de remoção inesperado.'
    }
    Remove-Item -LiteralPath $resolvedSource -Recurse -Force
}
[ordered]@{ source=$sourcePath; destination=$destinationPath;
    status=$(if ($Acao -eq 'Copiar') { 'verified' } else { 'archived' });
    checked_at=[DateTimeOffset]::UtcNow.ToString('o'); files=$checks } |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statePath -Encoding UTF8
Write-Host $(if ($Acao -eq 'Copiar') { 'Cópia conferida. A origem ainda está preservada.' }
    else { 'Arquivamento concluído. A origem foi removida após a conferência.' })
