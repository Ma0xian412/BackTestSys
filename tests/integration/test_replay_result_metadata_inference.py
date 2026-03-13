"""Replay 策略结果元数据推断集成测试。"""

from __future__ import annotations

from quant_framework.adapters.factory import BacktestConfigFactory
from quant_framework.config import BacktestConfig
from quant_framework.core import BacktestApp


def _write_market_data_csv(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("RecvTick,ExchTick,Bid,BidVol,Ask,AskVol,LastVolSplit\n")
        f.write('1000,1000,100.0,100,101.0,100,"[(101.0, 60)]"\n')
        f.write('2000,2000,100.0,80,101.0,90,"[(101.0, 40)]"\n')


def _write_replay_order_csv(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "PartitionDay,ContractId,MachineName,OrderId,LimitPrice,Volume,OrderDirection,SentTime\n"
        )
        f.write("20250102,12345,replay-node-a,1,200.0,10,Buy,1000\n")


def _write_replay_cancel_csv(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("PartitionDay,ContractId,MachineName,OrderId,CancelSentTime\n")
        f.write("20250102,12345,replay-node-a,1,1500\n")


def test_replay_result_metadata_inferred_from_replay_csv(tmp_path):
    market_data_file = str(tmp_path / "market.csv")
    order_file = str(tmp_path / "orders.csv")
    cancel_file = str(tmp_path / "cancels.csv")
    _write_market_data_csv(market_data_file)
    _write_replay_order_csv(order_file)
    _write_replay_cancel_csv(cancel_file)

    config = BacktestConfig()
    config.data.format = "csv"
    config.data.path = market_data_file
    config.strategy.name = "ReplayStrategy_Impl"
    config.strategy.params.order_file = order_file
    config.strategy.params.cancel_file = cancel_file
    config.contract.machine_name = ""
    config.contract.contract_info = None

    runtime_cfg = BacktestConfigFactory().create(config)
    result = BacktestApp(runtime_cfg).run()

    assert len(result.DoneInfo) == 1
    assert len(result.OrderInfo) == 1
    assert len(result.ExecutionDetail) == 1
    assert len(result.CancelRequest) == 1

    expected_day = 20250102
    expected_contract_id = "12345"
    expected_machine = "replay-node-a"

    for row in result.DoneInfo:
        assert row.PartitionDay == expected_day
        assert row.ContractId == expected_contract_id
        assert row.MachineName == expected_machine
    for row in result.OrderInfo:
        assert row.PartitionDay == expected_day
        assert row.ContractId == expected_contract_id
        assert row.MachineName == expected_machine
    for row in result.ExecutionDetail:
        assert row.PartitionDay == expected_day
        assert row.ContractId == expected_contract_id
        assert row.MachineName == expected_machine
    for row in result.CancelRequest:
        assert row.PartitionDay == expected_day
        assert row.ContractId == expected_contract_id
        assert row.MachineName == expected_machine
