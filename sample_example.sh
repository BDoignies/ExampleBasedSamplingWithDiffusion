output_dir=tmp # Output dir of the script
model=LDBN     # Name of the model (see: https://projet.liris.cnrs.fr/qmcdiffusion/models_/)   
               # Warning: this script does not work with 'cond'

mkdir -p $output_dir

# Get models data (if it does not exist first)
if [ ! -f $output_dir/$model.ckpt ] 
then
    # Curl is required for this script. Most of the time already installed, but in case: (sudo apt install curl)
    curl https://projet.liris.cnrs.fr/qmcdiffusion/models_/$model/model.ckpt  --output $output_dir/$model.ckpt
    curl https://projet.liris.cnrs.fr/qmcdiffusion/models_/$model/config.json --output $output_dir/$model.json
fi

# Sample 1 example from model (1024 pts (= 32 * 32), using 100 timesteps)
python sample.py -c $output_dir/$model.json -m $output_dir/$model.ckpt -s 1 2 32 32 -t 100 -o $output_dir/$model

# Plot pts
plot_code="import matplotlib.pyplot as plt; import numpy as np; data = np.load(\"$output_dir/$model.npy\")[0]; plt.gca().set_aspect(\"equal\"); plt.scatter(*data.T, s=2); plt.show()"
python -c "$plot_code"