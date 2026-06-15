"""Тесты Payload-эксперта: токенизация (PAD/UNK/truncate), char-CNN."""

import torch

from mlsiem.models.payload.model import (
    PAD_IDX,
    UNK_IDX,
    CharCNN,
    CharCNNConfig,
    CharTokenizer,
)


def test_tokenizer_pad_and_truncate():
    tok = CharTokenizer.fit(["abcabc"], max_len=4)
    enc = tok.encode("ab")
    assert enc.tolist()[2:] == [PAD_IDX, PAD_IDX]   # хвост паддинг
    assert tok.encode("abcdef").shape == (4,)        # обрезка до max_len


def test_tokenizer_unk_for_unseen_char():
    tok = CharTokenizer.fit(["abc"], max_len=5)
    enc = tok.encode("aZ")        # Z не виден при fit
    assert enc[0] != UNK_IDX      # 'a' известен
    assert enc[1] == UNK_IDX      # 'Z' → UNK


def test_tokenizer_roundtrip(tmp_path):
    tok = CharTokenizer.fit(["GET /x?q=1"], max_len=16)
    loaded = CharTokenizer.load(tok.save(tmp_path / "tok.json"))
    assert loaded.chars == tok.chars
    assert loaded.encode("GET").tolist() == tok.encode("GET").tolist()


def test_charcnn_forward_shape():
    tok = CharTokenizer.fit(["abcdef" * 5], max_len=32)
    model = CharCNN(CharCNNConfig(vocab_size=tok.vocab_size, n_classes=2))
    x = torch.randint(0, tok.vocab_size, (8, 32))
    out = model(x)
    assert out.shape == (8, 2)


def test_charcnn_backward():
    model = CharCNN(CharCNNConfig(vocab_size=50, n_classes=2))
    x = torch.randint(0, 50, (4, 20))
    loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 0, 1]))
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_padding_idx_embedding_is_zero():
    model = CharCNN(CharCNNConfig(vocab_size=50, n_classes=2))
    pad_emb = model.embedding.weight[PAD_IDX]
    assert torch.allclose(pad_emb, torch.zeros(model.embedding.embedding_dim))
