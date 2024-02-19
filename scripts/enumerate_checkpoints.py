#!/usr/bin/env python3

import argparse
import collections
import torch
import os
import re


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of checkpoints to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    for f in inputs:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state['model']

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            if k not in params_dict:
                params_dict[k] = []
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            params_dict[k].append(p)

    averaged_params = collections.OrderedDict()
    # v should be a list of torch Tensor.
    for k, v in params_dict.items():
        summed_v = None
        for x in v:
            summed_v = summed_v + x if summed_v is not None else x
        averaged_params[k] = summed_v / len(v)
    new_state['model'] = averaged_params
    return new_state


def enumeration_checkpoints(paths, start_avg_num, upper_bound, lower_bound):
    assert len(paths) == 1
    path = paths[0]

    pt_regexp = re.compile(r'checkpoint(\d+)\.pt')
    files = os.listdir(path)

    entries = []
    for f in files:
        m = pt_regexp.fullmatch(f)
        if m is not None:
            sort_key = int(m.group(1))
            if sort_key >= lower_bound and sort_key <= upper_bound:
                entries.append((sort_key, m.group(0)))

    entries = sorted(entries)
    len_entries = len(entries)
    if entries[0][0] != lower_bound:
        raise Exception('can not find file checkpoint{}.pt ', lower_bound)
    if entries[-1][0] != upper_bound: 
        raise Exception('can not find file checkpoint{}.pt ', upper_bound)
    if len_entries != (upper_bound - lower_bound +1):
        raise Exception('the checkpoints seems not contigurous')
    #return [os.path.join(path, x[1]) for x in sorted(entries)]

    assert start_avg_num <= len_entries

    result = []
    for i in range(start_avg_num, len_entries+1):
        for j in range(0, len_entries+1-i):
            result.append(entries[j:j+i])
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Tool to average the params of input checkpoints to '
                    'produce a new checkpoint',
    )
    # fmt: off
    parser.add_argument('--inputs', required=True, nargs='+',
                        help='Input checkpoint file paths.')
    parser.add_argument('--output', required=True, type=str,
                        help='Write the new checkpoint containing the averaged weights to this path.')
    #num_group = parser.add_mutually_exclusive_group()
    #num_group.add_argument('--num-epoch-checkpoints', type=int,
    #                       help='if set, will try to find checkpoints with names checkpoint_xx.pt in the path specified by input, '
    #                       'and average last this many of them.')
    #num_group.add_argument('--num-update-checkpoints', type=int,
    #                       help='if set, will try to find checkpoints with names checkpoint_ee_xx.pt in the path specified by input, '
    #                       'and average last this many of them.')
    parser.add_argument('--checkpoint-upper-bound', type=int, help='checkpoint-upper-bound')
    parser.add_argument('--checkpoint-lower-bound', type=int, help='checkpoint-lower-bound')
    parser.add_argument('--start-avg-num', default=2, type=int, help='start average num ')
    # fmt: on
    args = parser.parse_args()
    print(args)


    assert args.checkpoint_upper_bound is not None and args.checkpoint_lower_bound is not None, \
            'requires --checkpoint-upper-bound and --checkpoint-lower-bound'
    #assert args.num_epoch_checkpoints is None or args.num_update_checkpoints is None, \
    #        'Cannot combine --num-epoch-checkpoints and --num-update-checkpoints'

    enumeration_result = enumeration_checkpoints(
            args.inputs, args.start_avg_num, args.checkpoint_upper_bound, args.checkpoint_lower_bound
        )

    path = args.inputs[0]
    for combine in enumeration_result:
        ckpts = [os.path.join(path, x[1]) for x in combine]
        output = "checkpoint{}-{}.pt".format(combine[0][0], combine[-1][0])
        new_state = average_checkpoints(ckpts)
        torch.save(new_state, args.output+'/'+output)
    print('Finished writing averaged checkpoint to {}.'.format(args.output))


if __name__ == '__main__':
    main()
