def sample_data(tmp_path):
    """Create sample dataset for testing."""
    data = {
        'text': [
            'قال شهريار للوزير',
            'في مدينة بغداد القديمة',
            'وجد علاء الدين المصباح السحري'
        ],
        'labels': [
            'O B-PERSON O',
            'O B-LOCATION I-LOCATION O',
            'O B-PERSON I-PERSON B-OBJECT I-OBJECT'
        ]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture
def dataset(sample_data):
    """Create dataset instance for testing."""
    return ArabicNERDataset(sample_data)

def test_dataset_loading(dataset):
    """Test dataset loading and basic properties."""
    assert len(dataset) == 3
    assert len(dataset.texts) == len(dataset.labels)
    assert isinstance(dataset[0], dict)
    assert 'tokens' in dataset[0]
    assert 'labels' in dataset[0]

def test_label_distribution(dataset):
    """Test label distribution calculation."""
    distribution = dataset.get_label_distribution()
    assert distribution['O'] == 4
    assert distribution['B-PERSON'] == 2
    assert distribution['B-OBJECT'] == 1

def test_dataset_splitting(dataset):
    """Test dataset splitting functionality."""
    train, val, test = dataset.split_dataset(0.6, 0.2, seed=42)
    assert len(train) + len(val) + len(test) == len(dataset)
    assert isinstance(train, ArabicNERDataset)
    assert isinstance(val, ArabicNERDataset)
    assert isinstance(test, ArabicNERDataset)

def test_max_length_handling(sample_data):
    """Test handling of sequences longer than max_length."""
    dataset = ArabicNERDataset(sample_data, max_length=2)
    sample = dataset[0]
    assert len(sample['tokens']) == 2
    assert len(sample['labels']) == 2

def test_label_mapping(dataset):
    """Test label mapping functionality."""
    sample = dataset[0]
    assert isinstance(sample['labels'], torch.Tensor)
    assert sample['labels'].dtype == torch.long

def test_invalid_data_file():
    """Test handling of invalid data file."""
    with pytest.raises(FileNotFoundError):
        ArabicNERDataset("nonexistent.csv")

def test_padding(sample_data):
    """Test sequence padding."""
    dataset = ArabicNERDataset(sample_data, max_length=5)
    sample = dataset[0]
    assert len(sample['tokens']) == 5
    assert len(sample['labels']) == 5

def test_custom_label_map(sample_data):
    """Test custom label mapping."""
    custom_map = {
        'O': 0,
        'B-PERSON': 1,
        'I-PERSON': 2,
        'B-LOCATION': 3,
        'I-LOCATION': 4,
        'B-OBJECT': 5,
        'I-OBJECT': 6
    }
    dataset = ArabicNERDataset(sample_data, label_map=custom_map)
    sample = dataset[0]
    assert max(sample['labels']).item() < len(custom_map)
