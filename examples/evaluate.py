import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from modelvshuman import Plot, Evaluate
from modelvshuman import constants as c
from plotting_definition import plotting_definition_template


def run_evaluation():
    # models = ["resnet50", "bagnet33", "simclr_resnet50x1"]
    # models = ["deit_base_patch16_224", "deit_small_patch16_224", "deit_tiny_patch16_224", "resnet50", "resnet50_trained_on_SIN", "resnet50_trained_on_SIN_and_IN"]
    models = ["deit_tiny_no_positional_encoding", "deit_tiny_patch16_224", "jigsaw_tiny_patch16_224"]
    datasets = c.DEFAULT_DATASETS # or e.g. ["cue-conflict", "uniform-noise"]
    params = {"batch_size": 64, "print_predictions": True, "num_workers": 20}
    Evaluate()(models, datasets, **params)


def run_plotting():
    plot_types = c.DEFAULT_PLOT_TYPES # or e.g. ["accuracy", "shape-bias"]
    # plot_types = ["shape-bias"]
    plotting_def = plotting_definition_template
    figure_dirname = "example-figures/"
    Plot(plot_types = plot_types, plotting_definition = plotting_def,
         figure_directory_name = figure_dirname)

    # In examples/plotting_definition.py, you can edit
    # plotting_definition_template as desired: this will let
    # the toolbox know which models to plot, and which colours to use etc.


if __name__ == "__main__":
    # 1. evaluate models on out-of-distribution datasets
    run_evaluation()
    # 2. plot the evaluation results
    run_plotting()
