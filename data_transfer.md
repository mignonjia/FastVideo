Here is the commands to load data

## Login (required for private repos)

huggingface-cli login

## Download a dataset (repo) from the command line

### Download to current directory (creates a folder named after the repo)
huggingface-cli download mignonjia/zelda_overfit4 --repo-type dataset

### Download to a specific local folder
huggingface-cli download mignonjia/zelda_overfit4 --repo-type dataset --local-dir /home/hal-shared/world_model/zelda_overfit4

### Download only .tar.gz files
huggingface-cli download mignonjia/traindata_0205_1330 --repo-type dataset --local-dir /home/hal-shared/world_model/MC --include "*.tar.gz"


## Extract after download

### .tar.gz (e.g. from --include "*.tar.gz")
tar -xzf traindata_0205_1330.tar.gz -C /home/hal-shared/world_model/MC

tar -xzf traindata_0208_2000.tar.gz -C /home/hal-shared/world_model/MC
