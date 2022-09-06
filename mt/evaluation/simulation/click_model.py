import numpy as np
from abc import ABC, abstractmethod


class ClickModel(ABC):
    
    def __init__(
        self, 
        label_list: list,
        max_rank: int,
        click_noise: float = 0.1,
        control_pb_severe: float = 0.5
     ) -> None:
        self.click_probs = self._set_click_probs(label_list, click_noise)
        self.exam_probs = self._set_examination_probs(max_rank, control_pb_severe)

    @staticmethod
    def _set_click_probs(label_list, eta: float = 0.1):
        y_max = max(label_list)
        click_probs = [
            eta + (1-eta) * ( (pow(2, y) - 1) / (pow(2, y_max) - 1) )
            for y in sorted(label_list, reverse=False)
        ]
        return click_probs

    @abstractmethod
    def _set_examination_probs(max_rank, nu):
        return [(1/(i+1))**nu for i in range(max_rank)]

    @abstractmethod
    def sim_clicks_for_serp(self, serp):
        pass


class SimPosBias(ClickModel):

    def __init__(self, label_list: list, max_rank: int, click_noise: float = 0.1, control_pb_severe: float = 1, layout_types=1) -> None:
        
        self.layout_types = layout_types
        super().__init__(label_list, max_rank, click_noise, control_pb_severe)
    
    
    def sim_clicks_for_serp(self, serp):
        click_list = []

        for pos, product in enumerate(serp.relevances):
            # get P(E=1)
            if serp.layout_type is not None:
                exam_prob = self.exam_probs[serp.layout_type][pos]
                is_last = pos == len(self.exam_probs[serp.layout_type])
            else:
                exam_prob = self.exam_probs[pos]
                is_last = pos == len(self.exam_probs)
            # get P(R=1)
            click_prob = self.click_probs[product]
            # observe C
            click = 1 if np.random.random() < (exam_prob*click_prob) else 0

            click_list.append(click)

            if is_last:
                break

        return click_list

    def _set_examination_probs(self, max_rank, nu):
        if self.layout_types == 1:
            return [(1/(i+1))**nu for i in range(max_rank)]
        else:
            assert len(nu) == self.layout_types, "specify nu for each layout type"
            return {
                l: [(1/(i+1))**nu[l] for i in range(max_rank)] for l in range(self.layout_types)
            } 


class SimClickAndPurchase(ClickModel):
    # TODO: purchase probability should increase strictly monotonically with
    # increasing relevance label, whereas click probability could be some threshold
    # if label in [0,3] click prob < 0.5 if label in [4,5] click_prob = 1
    # just think of something smart, which makes sense and a model can learn
    def __init__(
        self, 
        label_list: list, 
        max_rank: int, 
        click_noise: float = 0.1, 
        control_pb_severe: float = 1
    ) -> None:
        super().__init__(label_list, max_rank, click_noise, control_pb_severe)
        self.buy_prob = self._set_purchase_probs(label_list, .5, 0)

    @staticmethod
    def _set_purchase_probs(label_list, pos_click_prob=1.0, neg_click_prob=0.0):
        b = (pos_click_prob - neg_click_prob) / \
            (pow(2, max(label_list)) - 1)
        a = neg_click_prob - b
        return [a + pow(2, i)*b for i in range(len(label_list))]

    def sim_clicks_for_serp(self, serp):
        click_list = []
        for pos, product in enumerate(serp.relevances):
            exam_prob = self.exam_probs[pos]
            click_prob = self.click_probs[product]
            click = 1 if np.random.random() < (exam_prob * click_prob) else 0
            if click == 1:
                purchase = np.random.random() < self.buy_prob[product]
            else:
                purchase = 0

            click_list.append(click + purchase)
            if pos >= len(self.exam_probs) -1:
                break

        return click_list
