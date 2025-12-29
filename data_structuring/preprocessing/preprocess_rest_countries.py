import orjson
import zlib

from data_structuring.config import DatabaseConfig, PreprocessRestCountriesCLIConfig

DOMAIN_EXTENSIONS_KEY = "domain_extensions"
POSTAL_CODE_REGEX_KEY = "postal_code_regex"
PHONE_PREFIXES_KEY = "phone_prefixes"

if __name__ == "__main__":
    # Parse CLI args
    cli_args = PreprocessRestCountriesCLIConfig()
    # Load database config
    config = DatabaseConfig()

    with open(cli_args.input_rest_countries_path, "r") as file:
        restCountries = orjson.loads(file.read())

    with open(config.country_specs, "wb") as f:
        f.write(
            zlib.compress(
                orjson.dumps({
                    country["cca2"].upper(): {
                        DOMAIN_EXTENSIONS_KEY: country["tld"],
                        POSTAL_CODE_REGEX_KEY: country["postalCode"]["regex"],
                        PHONE_PREFIXES_KEY: [country["idd"]["root"] + country["idd"]["suffixes"][i]
                                             for i in range(len(country["idd"]["suffixes"]))]
                    }
                    for country in restCountries
                })))
