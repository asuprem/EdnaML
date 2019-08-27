import os, shutil, logging, glob, re, pdb, json
import kaptan
import click
import utils
import torch, torchsummary




@click.command()
@click.argument('config')
@click.option('--mode', default="train", help="Execution mode: [train/test]")
@click.option('--weights', default=".", help="Path to weights if mode is test")
def main(config, mode, weights):
    pass


if __name__ == "__main__":
    main()