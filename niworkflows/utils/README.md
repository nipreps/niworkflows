# niworkflows.utils.testing
## Generating a BIDS skeleton
Creating a BIDS skeleton can be useful when testing methods that operate on diverse BIDS datasets.
This readme contains some information on using `niworkflows.utils.tests.bids.generate_bids_skeleton()` to create a BIDS skeleton.


### Example configuration

> sessions.yaml

```yaml
dataset_description:
  Name: sessions
  BIDSVersion: 1.6.0
'01':
- session: pre
  anat:
  - suffix: T1w
    metadata:
      EchoTime: 1
  func:
  - task: rest
    echo: 1
    suffix: bold
    metadata:
      RepetitionTime: 0.8
      EchoTime: 0.5
      TotalReadoutTime: 0.5
      PhaseEncodingDirection: j
  - task: rest
    echo: 2
    suffix: bold
    metadata:
      RepetitionTime: 0.8
      EchoTime: 0.7
      TotalReadoutTime: 0.5
      PhaseEncodingDirection: j
- session: post
  anat:
    suffix: T2w
    metadata:
      EchoTime: 2
  func:
    task: rest
    acq: lowres
    suffix: bold
    metadata:
      RepetitionTime: 0.8
      PhaseEncodingDirection: j-
'02': "*"
'03': "*"
```


### Keys

#### Top level keys
| Key | Description | Required | Values |
| --- | ----------- | -------- | ------ |
| `dataset_description` | Top level JSON to describe the dataset | Optional | `metadata`* |
| `participant` | Participant ID (`sub` prefix not required) | Required | One or more `session`s or `*`**. |

>\* Metadata must include the following fields: `Name`, `BIDSVersion` \
\*\* The `*` will recursively copy all values from the `participant` above.

If `dataset_description` is not specified, a default will be created.

#### Other keys
| Key | Description | Required | Values |
| --- | ----------- | -------- | ------ |
| `session` | A logical grouping of data | Required | One or more `datatype`s + optional `session` field (if multi-session) |
| `datatype` | A functional group of similar data types | Required | One or more `filepair`s |
| `filepair` | Data and associated metadata | Required | BIDS `entities`* + optional `metadata` |
| `metadata` | Sidecar JSON values corresponding to the data file | Optional | Any BIDS field/value pairs (information is dumped to a JSON file) |


>\* The only required `entity` is `suffix`. Ordering is respected, so ensure field order respects BIDS specification.
