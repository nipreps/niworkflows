from copy import deepcopy
import json
from pathlib import Path
import yaml


def generate_bids_skeleton(target_path, bids_config):
    """
    Converts a BIDS directory in dictionary form to a file structure.

    The BIDS configuration can either be a YAML or JSON file, or :obj:dict: object.

    Parameters
    ----------
    target_path : str
        Path to generate BIDS directory at (must not exist)
    bids_config : dict or str
        Configuration on how to create the BIDS directory.
    """

    if isinstance(bids_config, dict):
        # ensure dictionary remains unaltered
        bids_dict = deepcopy(bids_config)
    elif isinstance(bids_config, str):
        bids_config = Path(bids_config).read_text()
        try:
            bids_dict = json.loads(bids_config)
        except json.JSONDecodeError:
            bids_dict = yaml.load(bids_config, Loader=yaml.Loader)

    _bids_dict = deepcopy(bids_dict)
    root = Path(target_path).absolute()
    root.mkdir(parents=True)

    desc = bids_dict.pop("dataset_description", None)
    if desc is None:
        # default description
        desc = {"Name": "Default", "BIDSVersion": "1.6.0"}
    to_json(root / "dataset_description.json", desc)

    cached_subject_data = None
    for subject, sessions in bids_dict.items():
        bids_subject = subject if subject.startswith("sub-") else f"sub-{subject}"
        subj_path = root / bids_subject
        subj_path.mkdir(exist_ok=True)

        if sessions == "*":  # special case to copy previous subject data
            sessions = cached_subject_data.copy()

        if isinstance(sessions, dict):  # single session
            sessions.update({"session": None})
            sessions = [sessions]

        cached_subject_data = deepcopy(sessions)
        for session in sessions:

            ses_name = session.pop("session", None)
            if ses_name is not None:
                bids_session = ses_name if ses_name.startswith("ses-") else f"ses-{ses_name}"
                bids_prefix = f"{bids_subject}_{bids_session}"
                curr_path = subj_path / bids_session
                curr_path.mkdir(exist_ok=True)
            else:
                bids_prefix = bids_subject
                curr_path = subj_path

            # create modalities
            for modality, files in session.items():
                modality_path = curr_path / modality
                modality_path.mkdir(exist_ok=True)

                if isinstance(files, dict):  # single file / metadata combo
                    files = [files]

                for bids_file in files:
                    metadata = bids_file.pop("metadata", None)
                    suffix = bids_file.pop("suffix")
                    entities = combine_entities(**bids_file)
                    nii_file = modality_path / f"{bids_prefix}{entities}_{suffix}.nii.gz"
                    nii_file.touch()

                    if metadata is not None:
                        nii_metadata = nii_file.parent / nii_file.name.replace("nii.gz", "json")
                        to_json(nii_metadata, metadata)

    return _bids_dict


def to_json(filename, data):
    filename = Path(filename)
    filename.write_text(json.dumps(data))
    return filename


def combine_entities(**entities):
    return f"_{'_'.join([f'{lab}-{val}' for lab, val in entities.items()])}" if entities else ""
