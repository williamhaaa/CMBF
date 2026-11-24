"""Exercise all three original CMBF branches, auxiliary losses and backward on CPU."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp.exp_main import Exp_Main  # Verify the complete original training import tree.
from models.EventMSHyperV1 import Model


def main():
    torch.manual_seed(2021)
    torch.set_num_threads(1)
    config = SimpleNamespace(
        seq_len=24, pred_len=12, enc_in=4, individual=True,
        t_cscm='Bottleneck_Construct', t_window_size=[2], t_hyper_num=[6, 3],
        t_d_model=8, t_kprime=3, t_beta_thr=0.0,
        event_layers=2, num_event_protos=8, proto_topk=4, event_topk=4,
        event_heads=4, dec_heads=4, dec_layers=1, dec_ffn_dim=16,
        use_cycle_index=True, cycle=24,
        use_linear=True, use_node=True, use_event=True,
        dropout=0.0, event_dropout=0.0, viz_auto_eval=False,
    )
    model = Model(config)
    inputs = torch.randn(2, config.seq_len, config.enc_in)
    marks = torch.zeros(2, config.seq_len, 4)
    cycles = torch.arange(config.seq_len).unsqueeze(0).expand(2, -1)
    model.train()
    prediction, auxiliary = model(inputs, marks, cycle_index=cycles)
    assert prediction.shape == (2, config.pred_len, config.enc_in), prediction.shape
    assert torch.isfinite(prediction).all()
    auxiliary_tensors = [v for v in auxiliary.values() if torch.is_tensor(v)]
    assert all(torch.isfinite(v).all() for v in auxiliary_tensors)
    loss = prediction.square().mean() + sum(v.mean() for v in auxiliary_tensors)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters())
               for branch in [model.Linear, model.time_hgconvs, model.event_tower_time])
    model.eval()
    with torch.no_grad():
        prediction_eval, _ = model(inputs, marks, cycle_index=cycles)
    assert prediction_eval.shape == prediction.shape and torch.isfinite(prediction_eval).all()
    print(json.dumps({'status': 'passed', 'model': 'EventMSHyperV1',
                      'branches': ['linear', 'node', 'event'],
                      'prediction_shape': list(prediction.shape),
                      'finite_backward': True, 'train_and_eval': True,
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'torch': torch.__version__}, indent=2))


if __name__ == '__main__':
    main()
