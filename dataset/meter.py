class TopKClassMeter:
    def __init__(self, k):
        self.reset()
        self.k = k

    def reset(self):
        self.num_examples = 0
        self.num_correct = 0

    def update(self, outputs, targets):
        _, indices = outputs.topk(self.k, 1, True, True)

        indices = indices.transpose(0, 1)
        masks = indices.eq(targets.view(1, -1).expand_as(indices))

        self.num_examples += targets.size(0)
        self.num_correct += masks[:self.k].reshape(-1).float().sum(0)

    def compute(self):
        return self.num_correct / max(self.num_examples, 1) * 100.

    def data(self):
        return {'num_examples': self.num_examples,
                'num_correct': self.num_correct}

    def set(self, data):
        if 'num_examples' in data:
            self.num_examples = data['num_examples']
        if 'num_correct' in data:
            self.num_correct = data['num_correct']


def make_meters():
    return {
        'acc/{}_top1': TopKClassMeter(1),
        'acc/{}_top5': TopKClassMeter(5)
    }
