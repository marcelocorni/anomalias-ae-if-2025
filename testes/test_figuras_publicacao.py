import pytest

from src.figuras_publicacao import validate_matrix


def result():
    return {'rows':100, 'positives':10, 'unassociated':90,
            'thresholds':{'q995':{'tn':85,'fp':5,'fn':3,'tp':7}}}


def test_matriz_tem_associacao_nas_linhas_e_sinalizacao_nas_colunas():
    assert validate_matrix(result(),'q995',12)==[[85,5],[3,7]]


def test_matriz_recusa_populacao_ou_sinalizacao_incompativel():
    r=result();r['rows']=101
    with pytest.raises(ValueError,match='população'):validate_matrix(r,'q995',12)
    with pytest.raises(ValueError,match='sinalização'):validate_matrix(result(),'q995',13)


def test_matriz_recusa_contagens_negativas():
    r=result();r['thresholds']['q995']['tp']=-1
    with pytest.raises(ValueError,match='inválida'):validate_matrix(r,'q995',12)
