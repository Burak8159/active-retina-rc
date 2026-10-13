from active_retina.train_every_image_autonomy import sampler_weights


def test_source_balanced_sampler_equalizes_source_mass_without_reweighting_labels():
    rows = [
        {"source_dataset": "large", "grade": 0, "referable_dr": 0},
        {"source_dataset": "large", "grade": 2, "referable_dr": 1},
        {"source_dataset": "large", "grade": 1, "referable_dr": 0},
        {"source_dataset": "small", "grade": 0, "referable_dr": 0},
    ]

    weights = sampler_weights(rows, 5.0, 5.0, 5.0, source_balanced=True)

    assert weights == [1 / 3, 1 / 3, 1 / 3, 1.0]
    assert sum(weights[:3]) == sum(weights[3:])


def test_natural_sampler_weights_preserve_existing_positive_boost():
    rows = [
        {"source_dataset": "a", "grade": 0, "referable_dr": 0},
        {"source_dataset": "b", "grade": 2, "referable_dr": 1},
    ]

    assert sampler_weights(rows, 1.0, 1.0, 1.0) == [1.0, 3.0]
