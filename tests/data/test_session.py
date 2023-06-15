import numpy as np

from cookie_test.data.session import Session


def test_session_from_path(mock_session_path, mock_responses):
    session = Session.from_path(mock_session_path)

    assert session.subject_id == "001"
    assert session.session_id == "123"
    assert (session.responses == mock_responses).all()
    assert (session.image_ids == np.array([1, 2, 3])).all()
    assert session.previous_image_ids is None
    assert session.trial_ids is None
