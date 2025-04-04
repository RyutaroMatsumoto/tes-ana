# Deactivate Anaconda 'base' environment
conda deactivate

# Verify the environment change
which python
python --version

# Activate your custom environment (Optional)
source /Users/ryutaro_matsumoto/Desktop/Reaserch/Analysis_codes/myenv/bin/activate

# Confirm that the intended Python interpreter from Pyenv is active
which python
python --version

# If encountering any issues, adjust your PATH environment variable manually
export PATH="/Users/ryutaro_matsumoto/Desktop/Reaserch/Analysis_codes/myenv/bin:$PATH"

# To make persistent changes and avoid repeating these steps in future sessions, add the necessary commands to your shell configuration file
echo 'export PATH="/Users/ryutaro_matsumoto/Desktop/Reaserch/Analysis_codes/myenv/bin:$PATH"' >> ~/.zshrc
