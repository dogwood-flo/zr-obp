from .real import OpenBanditDataset

from dataclasses import dataclass
from typing import Tuple, Union
from ..types import BanditFeedback


@dataclass
class OpenBanditDatasetWithTimestamp(OpenBanditDataset):
    include_timestamp: bool = True

    def load_raw_data(self):
        super().load_raw_data()
        self.timestamp = self.data["timestamp"].values

    def obtain_batch_bandit_feedback(
        self, test_size: float = 0.3, is_timeseries_split: bool = False
    ) -> Union[BanditFeedback, Tuple[BanditFeedback, BanditFeedback]]:
        if not isinstance(is_timeseries_split, bool):
            raise TypeError(
                f"`is_timeseries_split` must be a bool, but {type(is_timeseries_split)} is given"
            )
        
        if self.include_timestamp:
            print("Including timestamp in feedback.")

        if is_timeseries_split:
            bandit_feedback_train, bandit_feedback_test = super().obtain_batch_bandit_feedback(
                test_size=test_size, is_timeseries_split=is_timeseries_split
            )
            bandit_feedback_train['timestamp'] = self.timestamp[:bandit_feedback_train['n_rounds']]
            bandit_feedback_test['timestamp'] = self.timestamp[bandit_feedback_train['n_rounds']:]
            return bandit_feedback_train, bandit_feedback_test
        else:
            bandit_feedback = super().obtain_batch_bandit_feedback(
                is_timeseries_split=is_timeseries_split
            )
            bandit_feedback['timestamp'] = self.timestamp
            return bandit_feedback

