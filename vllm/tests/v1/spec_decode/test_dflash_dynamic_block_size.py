import pytest

from vllm.v1.spec_decode.dflash import update_block_size_from_acceptance

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("current_block_size", "avg_acceptance", "expected"),
    [
        (2, 0.9, 3),
        (16, 0.9, 16),
        (4, 0.4, 3),
        (2, 0.4, 2),
        (8, 0.7, 8),
    ],
)
def test_update_block_size_from_acceptance(current_block_size, avg_acceptance, expected):
    assert update_block_size_from_acceptance(
        current_block_size=current_block_size,
        avg_acceptance=avg_acceptance,
        min_block_size=2,
        max_block_size=16,
    ) == expected
