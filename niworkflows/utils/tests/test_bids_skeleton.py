import pytest
from bids import BIDSLayout

from .test_bids_skeleton import generate_bids_skeleton


bids_dir_sessions = {
    "dataset_description": {"Name": "sample", "BIDSVersion": "1.6.0"},
    "01": [  # composed of dictionaries, pertaining to sessions
        {
            "session": "pre",
            "anat": [{"suffix": "T1w", "metadata": {"EchoTime": 1}}],  # anatomical files
            "func": [  # bold files
                {
                    "task": "rest",
                    "echo": 1,
                    "suffix": "bold",
                    "metadata": {
                        "RepetitionTime": 0.8,
                        "EchoTime": 0.5,
                        "TotalReadoutTime": 0.5,
                        "PhaseEncodingDirection": "j",
                    },
                },
                {
                    "task": "rest",
                    "echo": 2,
                    "suffix": "bold",
                    "metadata": {
                        "RepetitionTime": 0.8,
                        "EchoTime": 0.7,
                        "TotalReadoutTime": 0.5,
                        "PhaseEncodingDirection": "j",
                    },
                },
            ],
        },
        {
            "session": "post",
            "anat": {"suffix": "T2w", "metadata": {"EchoTime": 2}},
            "func": {
                "task": "rest",
                "acq": "lowres",
                "suffix": "bold",
                "metadata": {"RepetitionTime": 0.8, "PhaseEncodingDirection": "j-"},
            },
        },
    ],
    "02": "*",
    "03": "*",
}

bids_dir_session_less = {
    "01": [  # composed of dictionaries, pertaining to sessions
        {
            "anat": {"suffix": "T1w", "metadata": {"EchoTime": 1}},
            "func": [  # bold files
                {
                    "task": "rest",
                    "echo": 1,
                    "suffix": "bold",
                    "metadata": {
                        "EchoTime": 0.5,
                        "TotalReadoutTime": 0.5,
                        "PhaseEncodingDirection": "j",
                    },
                },
                {
                    "task": "rest",
                    "echo": 2,
                    "suffix": "bold",
                    "metadata": {
                        "EchoTime": 0.7,
                        "TotalReadoutTime": 0.5,
                        "PhaseEncodingDirection": "j",
                    },
                },
            ],
        }
    ],
    "02": "*",
    "03": {
        "anat": {"suffix": "T1w", "metadata": {"EchoTime": 1}},
        "func": [  # bold files
            {
                "task": "diff",
                "echo": 1,
                "suffix": "bold",
                "metadata": {
                    "EchoTime": 0.5,
                    "TotalReadoutTime": 0.5,
                    "PhaseEncodingDirection": "j",
                },
            },
            {
                "task": "diff",
                "echo": 2,
                "suffix": "bold",
                "metadata": {
                    "EchoTime": 0.7,
                    "TotalReadoutTime": 0.5,
                    "PhaseEncodingDirection": "j",
                },
            },
        ],
    },
    "04": "*",
}


@pytest.mark.parametrize(
    "json_layout,n_files,n_subjects,n_sessions",
    [
        (bids_dir_sessions, 31, 3, 2),
        (bids_dir_session_less, 25, 4, 0),
    ],
)
def test_generate_bids_skeleton(tmp_path, json_layout, n_files, n_subjects, n_sessions):
    generate_bids_skeleton(tmp_path, json_layout)
    datadesc = tmp_path / "dataset_description.json"
    assert datadesc.exists()
    assert "BIDSVersion" in datadesc.read_text()

    assert len([x for x in tmp_path.glob("**/*") if x.is_file()]) == n_files

    # ensure layout is valid
    layout = BIDSLayout(tmp_path)
    assert len(layout.get_subjects()) == n_subjects
    assert len(layout.get_sessions()) == n_sessions

    anat = layout.get(suffix="T1w", extension="nii.gz")[0]
    bold = layout.get(suffix="bold", extension="nii.gz")[0]
    assert anat.get_metadata()
    assert bold.get_metadata()
