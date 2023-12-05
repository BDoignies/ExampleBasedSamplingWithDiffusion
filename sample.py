import numpy as np
import argparse
import torch

from data.Transforms import to_pointset_optimal_transport

from utils.Config import ParseSampleConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Config file", required=True)
    parser.add_argument("-m", "--model", help="Path to checkpoint file", required=True)
    parser.add_argument("--cond", nargs='+', type=float, help='Conditionning values', required=False, default=None)
    parser.add_argument("-s", "--shape", nargs='+', type=int, help='Shape of points to generate', default=[16, 2, 8, 8])
    parser.add_argument("-o", "--output", help="Output file", default='out.npy')
    parser.add_argument("-t", "--timesteps", help="Number of timesteps", type=int, default=1000)
    parser.add_argument("--ot", help="Applied inverse ot mapping transform to input", type=bool, default=True)
    

    args = parser.parse_args()

    device = torch.device('cuda')
    model = ParseSampleConfig(args.config)
    
    model.load_state_dict(torch.load(args.model)["diffu"])
    model.to(device)
    model.set_num_timesteps(args.timesteps)
    model.eval()

    cond = args.cond
    if cond is not None:
        cond = torch.from_numpy(np.asarray(cond).astype(np.float32)).to(device)
        cond = cond.repeat(args.shape[0], 1)

    print(args.shape)
    with torch.no_grad():
        samples_tmp = model.p_sample_loop(args.shape, img=None, cond=cond, with_tqdm=True, with_sampling=True)
    
    samples = []
    samples_tmp = samples_tmp.cpu().numpy()

    if args.ot:
        for sample in samples_tmp:
            sample = to_pointset_optimal_transport(sample)
            samples.append(sample.reshape(sample.shape[0], np.prod(sample.shape[1:])).T)
            
        np.save(args.output, samples)
    else:
        np.save(args.output, samples_tmp)