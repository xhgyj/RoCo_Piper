from rocobrick.verification import ObservableConfirmation, TruthValidation, assembly_confirmed, final_truth_valid


def test_assembly_requires_connection_and_release_confirmation():
    assert not assembly_confirmed(ObservableConfirmation(True, True, True, False, False, 0.01))
    assert assembly_confirmed(ObservableConfirmation(True, True, True, True, False, 0.01))


def test_truth_requires_two_seconds_stability_and_no_unexpected_contact():
    assert not final_truth_valid(TruthValidation(True, True, 1.9, ()))
    assert final_truth_valid(TruthValidation(True, True, 2.0, ()))
