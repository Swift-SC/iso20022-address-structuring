"""
This module provides an interface to running pipeline for address structuring.
"""
import logging
import logging.config
import sys
# Standard libraries
import warnings

import orjson

from data_structuring.components.readers.base_reader import DEFAULT_ADDRESS_COLUMN, DEFAULT_SUGGESTED_COUNTRY_COLUMN, \
    DEFAULT_FORCE_SUGGESTED_COUNTRY_COLUMN

# Ignore the specific nested tensors warning from PyTorch
warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage and will change in the near future."
)

from data_structuring.components.readers.file_reader import TextFileReader, CsvFileReader
# Runners
from data_structuring.components.runners import ResultPostProcessing
from data_structuring.pipeline import AddressStructuringPipeline
from data_structuring.config import PostProcessingConfig, DEFAULT_LOGGING_CONFIG, RunCLIConfig

logger = logging.getLogger(__name__)


def _cli():
    """
    Function called when the program is used in CLI.
    Not meant to be used in any other way.
    """

    # Parse CLI args
    cli_args = RunCLIConfig()

    if cli_args.logging_config:
        with cli_args.logging_config.open() as f:
            json_config = orjson.loads(f.read())
            logging.config.dictConfig(json_config)
    else:
        logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
        logger.info("Default logging configuration used")

    # Parse input file
    logger.info("Parsing file %s", cli_args.input_data_path)
    if cli_args.input_data_path.suffix == ".txt":
        reader = TextFileReader(cli_args.input_data_path)
    elif cli_args.input_data_path.suffix in (".csv", ".tsv"):
        reader = CsvFileReader(cli_args.input_data_path,
                               sep=("\t" if cli_args.input_data_path.suffix == ".tsv" else ","),
                               data_column_name=DEFAULT_ADDRESS_COLUMN,
                               suggested_country_column=DEFAULT_SUGGESTED_COUNTRY_COLUMN,
                               force_suggested_country_column=DEFAULT_FORCE_SUGGESTED_COUNTRY_COLUMN)
    else:
        logger.error("Unsupported file format %s. Please provide a .txt, .csv or .tsv", cli_args.input_data_path)
        sys.exit(1)

    logger.info("Loading configuration")

    post_processing_config = PostProcessingConfig()
    # Create the DataStructuring object and perform inference
    ds = AddressStructuringPipeline()
    logger.info("Running inference on input")
    results = ds.run(reader, batch_size=cli_args.batch_size)

    if cli_args.output_data_path.suffix in (".csv", ".tsv"):
        _, saved_path = (
            ResultPostProcessing.save_list_as_human_readable_csv(
                results,
                file_name=cli_args.output_data_path,
                show_inferred_country=post_processing_config.show_inferred_country,
                verbose=cli_args.verbose))
    elif cli_args.output_data_path.suffix == ".json":
        # Exclude verbose fields if not in debugging mode
        exclude = ({"__all__": {"crf_result": {"emissions_per_tag", "log_probas_per_tag"}}}
                   if not cli_args.verbose
                   else None)
        saved_path = (
            ResultPostProcessing.save_list_as_json(
                results,
                file_name=cli_args.output_data_path,
                indent=2,
                exclude=exclude))
    else:
        logger.error("Unsupported output format: %s", cli_args.output_data_path.suffix)
        sys.exit(1)

    logger.info("ISO pipeline results saved in %s", saved_path)


if __name__ == "__main__":
    _cli()
