# 1. Tell Poetry globally: never create new venvs, always reuse existing ones
``` poetry config virtualenvs.create false ```

# this creates a .venv symlink in each project
``` poetry config virtualenvs.in-project true ```

# 2. Point every future project to your golden venv with a symlink

``` ln -s /home/ness/FuseMachines/audio/speech_seperation_pytorch/audio_sep_pytorch \
      ~/venvs/audio_sep_pytorch ```


# For every new project

cd ~/FuseMachines/audio/any_new_project

# Initialize Poetry (creates pyproject.toml)
poetry init --no-interaction

# Create the magic .venv symlink that points to your golden venv
ln -s ~/venvs/audio_sep_pytorch .venv

# Now install/add packages — they go straight into the golden venv
poetry install   # or poetry add torch jupyterlab etc.

-----------------------------------------------------------------------------------------
This finally worked
---------------------------------------------------------------------------------------
# 1. Fix the central location (if not already done)
mkdir -p ~/venvs
rm -rf ~/venvs/audio_sep_pytorch  # just in case
ln -s /home/ness/FuseMachines/audio/speech_seperation_pytorch/audio_sep_pytorch ~/venvs/audio_sep_pytorch

# 2. In your project folder — remove wrong/broken .venv if exists
cd ~/FuseMachines/audio/speech_seperation_nbc
rm -rf .venv

# 3. Create the correct symlink (note the "s" in venvs!)
ln -s ~/venvs/audio_sep_pytorch .venv

# 4. Done. That’s literally it.