# The Swift AI address structuring model

The address structuring model, aims to assist the community with the transition from unstructured postal addresses to
structured ISO 20022 CBPR+ format with field options for Town and Country. The model itself, although it does not
convert an unstructured address into a structured one, extracts the town and country (if present) from the given input
address.

This software solution uses a Conditional Random Field model alongside several fuzzy-matching and rule-based mechanisms
to infer structured the town and country.

## Quick start guide

### Pre-requisites

Before installing, ensure the following prerequisites are met:

- 4 GB of RAM
- Python 3.12 or higher
- pip (Python package installer)
- System compatible with PyTorch 2.6.0
- Swift Address Structuring model codebase (content of this repository)
- Swift Address Structuring model resources ([link](https://github.com/Swift-SC/iso20022-address-structuring-resources))
- Access to GeoNames to fetch the necessary files from the GeoNames export FTP server

### Environment Setup

Create the Python virtual environment (assuming Python 3.12):

```bash
python3 -m venv env
source env/bin/activate
``` 

Then download the required dependencies using pip
(specifying the **TMPDIR** is to avoid space issues during the installation process):

```bash
export TMPDIR=path/to/new/directory
python3 -m pip install -r requirements.txt
```

Finally, set the **PYTHONPATH** environment variable to the current directory:

```bash
export PYTHONPATH=$(pwd)
```

### Installing the reference datasets

Once the GeoNames files have been downloaded (refer to the User Documentation for links to the necessary files),
use the following scripts to generate the necessary reference dataset files:

```bash
# To install the towns and town aliases datasets
python3 data_structuring/preprocessing/preprocess_geonames_towns.py \
            --input_geonames_all_countries_path=path/to/allCountries.txt \
            --input_geonames_alternate_names_path=path/to/alternateNamesV2.txt \
            --input_geonames_country_info_path=path/to/countryInfo.txt

# To install the countries and country aliases datasets
python3 data_structuring/preprocessing/preprocess_geonames_countries.py \
            --input_geonames_all_countries_path=path/to/allCountries.txt \
            --input_geonames_alternate_names_path=path/to/alternateNamesV2.txt \
            --input_geonames_country_info_path=path/to/countryInfo.txt

# To install the postcodes datasets
python3 data_structuring/preprocessing/preprocess_geonames_postcodes.py \
            --input_geonames_postcodes_all_countries_path=path/to/postcodes/allCountries.txt \
            --input_geonames_postcodes_ca_full_path=path/to/postcodes/CA_full.txt \
            --input_geonames_postcodes_gb_full_path=path/to/postcodes/GB_full.txt \
            --input_geonames_postcodes_nl_full_path=path/to/postcodes/NL_full.txt

# To install the country_specs dataset
python3 scripts/preprocess_rest_countries.py \
            --input_rest_countries_path=path/to/countriesV3.1.json
```

**N.B.**: the input arguments can be *ignored* if all the downloaded files are put in the `resources/raw` folder as this
is the path used by default. In this case, the file structure should look like this:

```
resources
├── raw
│   ├── geonames
│   │   ├── allCountries.txt
│   │   ├── alternateNamesV2.txt
│   │   └── countryInfo.txt
│   ├── postcodes
│   │   ├── allCountries.txt
│   │   ├── CA_full.txt
│   │   ├── GB_full.txt
│   │   └── NL_full.txt
│   └── restCountries
│       └── countriesV3.1.json
```

### (Optional) Running the unit tests

This repository includes several unit tests to ensure the model works as intended.
The tests are run using the *pytest* Python testing framework and are executed with the following command:

```bash
python3 -m pytest tests
```

The tests may take some time to run as the *test_model.py* script runs the full model to ensure
that the model maintains an adequate level of performance.

### Usage

Running the model can be run on the provided input CSV file by using the following command:

```bash
python3 data_structuring/run.py \
            --input_data_path=data/input/addresses_gauntlet.csv \
            --verbose
```

This will generate an output file with the name *data_structuring_output.csv* with all the explainability columns
present.

### Leveraging country suggestion

It is possible to provide the model with country suggestions for certain addresses. In this case, the provided country
can be used in one of two ways:

- **Soft suggestion mode**: The suggested country will be added to the model's list of possible country suggestions (if
  it is not detected), and every possibility matching this country will receive a significant score boost (the score
  boost amount can be configured)
- **Hard suggestion mode**: Same as for the soft suggestion mode, however, the model's output is limited to
  possibilities located in this country

The suggested country should be stored in a column called "suggested_country" and can only be in the form of the
country's 2-letter ISO code or "NO COUNTRY" (if you wish to suggest that there should not be a country in the model's
output). The flag to enable either soft or hard suggestion mode should be stored in a column named "
force_suggested_country" and must be either "true", "1", or "yes".

Examples of input files enabling this feature can be found in the resources/input folder (look for files containing the
suffix "with_suggestions").

### Configuration and assessing model performance

Most parameters are controlled from the *config.py* file, or manually set up when creating the runners using the API.
Please refer to the more complete User Documentation for more.

In general, the default settings should provide satisfactory performance. There are nonetheless valid circumstances
where the model may under-perform. In these cases, experimentation is recommended and to aid with this, the provided
*addresses_gauntlet.csv* input file can be used to assess whether performance has been increased/decreased compared to
the baseline.

The tests/test_model.py script provides a simple way to run the model on the provided *addresses_gauntlet.csv* input
file (or any of the other input files) and calculate the performance of the model. The input file can be chosen by
modifying the script and uncommenting the required line under the *gauntlet_path* function (only one line should be
uncommented at any time):

```python
@pytest.fixture(autouse=True)
def gauntlet_path():
    return Path(resources.files(data_structuring.__name__) / ".." / "resources" / "input" / "addresses_gauntlet.csv")
    # return Path(resources.files(data_structuring.__name__) / ".." / "resources" / "input" / "addresses_gauntlet_with_suggestions.csv")
    # return Path(resources.files(data_structuring.__name__) / ".." / "resources" / "input" / "addresses_wikipedia.csv")
    # return Path(resources.files(data_structuring.__name__) / ".." / "resources" / "input" / "addresses_wikipedia_with_suggestions.csv")
```

The script can be run using the following command:

```bash
python3 -m pytest tests/test_model.py
```

The baselines model accuracy statistics are as follows:

```python
# addresses_gauntlet.csv
{
    # General accuracy
    'General country accuracy': 0.8530092592592593,
    'General town accuracy': 0.7858796296296297,
    'Combined general accuracy': 0.6944444444444444,
    # Specific accuracy scores
    'Correct country (present) accuracy': 0.65625,
    'Correct town (present) accuracy': 0.7164351851851852,
    'Correct country (not present) accuracy': 0.19675925925925927,
    'Correct town (not present) accuracy': 0.06944444444444445,
    # Statistics about the dataset
    'Number of countries (present)': 642,
    'Number of towns (present)': 769,
    'Number of countries (not present)': 222,
    'Number of towns (not present)': 95
}
# addresses_gauntlet.csv with country suggestion
{
    # General accuracy
    'General country accuracy': 1.0,
    'General town accuracy': 0.7013888888888888,
    'Combined general accuracy': 0.7013888888888888,
    # Specific accuracy scores
    'Correct country (present) accuracy': 1.0,
    'Correct town (present) accuracy': 0.6840052015604682,
    'Correct country (not present) accuracy': 1.0,
    'Correct town (not present) accuracy': 0.8421052631578947,
    # Statistics about the dataset
    'Number of countries (present)': 642,
    'Number of towns (present)': 769,
    'Number of countries (not present)': 222,
    'Number of towns (not present)': 95
}
# Wikipedia dataset
{
    # General accuracy
    'General country accuracy': 0.8358974358974359,
    'General town accuracy': 0.5948717948717949,
    'Combined general accuracy': 0.517948717948718,
    # Specific accuracy scores
    'Correct country (present) accuracy': 0.6358974358974359,
    'Correct town (present) accuracy': 0.5948717948717949,
    'Correct country (not present) accuracy': 0.2,
    'Correct town (not present) accuracy': 0.0,
    # Statistics about the dataset
    'Number of countries (present)': 132,
    'Number of towns (present)': 195,
    'Number of countries (not present)': 63,
    'Number of towns (not present)': 0
}
# Wikipedia with country suggestion dataset
{
    # General accuracy
    'General country accuracy': 1.0,
    'General town accuracy': 0.5948717948717949,
    'Combined general accuracy': 0.5948717948717949,
    # Specific accuracy scores
    'Correct country (present) accuracy': 1.0,
    'Correct town (present) accuracy': 0.5948717948717949,
    'Correct country (not present) accuracy': 0.0,
    'Correct town (not present) accuracy': 0.0,
    # Statistics about the dataset
    'Number of countries (present)': 195,
    'Number of towns (present)': 195,
    'Number of countries (not present)': 0,
    'Number of towns (not present)': 0
}
```