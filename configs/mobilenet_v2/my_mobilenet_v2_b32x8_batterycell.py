_base_ = 'mobilenet-v2_8xb32_in1k.py'

_deprecation_ = dict(
    expected='mobilenet-v2_8xb32_in1k.py',
    reference='https://github.com/open-mmlab/mmclassification/pull/508',
)
evaluation = dict(interval=1, metric='accuracy', save_best='auto')