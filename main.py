import argparse
import constants as const
from plotting import plot_images_and_labels
from constants import *
from train_model import train_model
from evaluate_model import compute_model_accuracy, display_model_predictions

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate a U-NET model for satellite imagery segmentation.")

    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument('--images-dir', type=str, required=True,
                            help="Path to the directory containing input images.")
    data_group.add_argument('--labels-dir', type=str, required=True,
                            help="Path to the directory containing ground truth labels.")
    data_group.add_argument('--train-size', type=float, default=TRAIN_SIZE,
                            help="Fraction of the dataset to use for training (default: 0.9).")

    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument('--input-shape', type=int, nargs=3, default=INPUT_SHAPE,
                             help="Input shape of the images as (height, width, channels). Default: (256, 256, 12).")

    # Training arguments
    training_group = parser.add_argument_group("Training")
    training_group.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                                help="Batch size for training and evaluation (default: 32).")
    training_group.add_argument('--epochs', type=int, default=EPOCHS,
                                help="Number of training epochs (default: 60).")
    training_group.add_argument('--save-model', action='store_true',
                                help="Specify whether to save the trained model (default: true).")
    training_group.add_argument('--no-save-model', action='store_false', dest='save_model',
                                help="Disable saving the trained model.")
    training_group.set_defaults(save_model=True)
    training_group.add_argument('--plot-training-summary', action='store_true',
                                help="Specify whether to plot the training summary (default: true).")
    training_group.add_argument('--no-plot-training-summary', action='store_false', dest='plot_training_summary',
                                help="Disable plotting the training summary.")
    training_group.set_defaults(plot_training_summary=True)
    training_group.add_argument('--save-training-summary', action='store_true',
                                help="Specify whether to save the training summary to a file (default: false).")

    # Output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument('--plot-images', action='store_true',
                              help="Specify whether to plot sample images and labels before training (default: false).")
    output_group.add_argument('--num-samples', type=int, default=3,
                              help="Number of sample images to plot (default: 3).")
    output_group.add_argument('--save-samples', action='store_true',
                              help="If not specified, samples will not be saved.")

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Log parsed arguments
    print(f"Arguments: {args}")

    # Optionally plot images and labels
    if args.plot_images:
        plot_images_and_labels(args.num_samples, args.save_samples, args.images_dir, args.labels_dir)

    # Train the model
    model_name = train_model(
        input_shape=tuple(args.input_shape),
        filters=const.FILTERS,
        n_classes=const.N_CLASSES,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_size=args.train_size,
        save_model=args.save_model,
        plot_training_summary=args.plot_training_summary,
        save_training_summary=args.save_training_summary,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir
    )

    # Compute model accuracy
    compute_model_accuracy(model_name, args.batch_size, args.train_size, args.images_dir, args.labels_dir)

    # Display model predictions
    display_model_predictions(model_name, args.batch_size, args.train_size)        


if __name__ == "__main__":
    main()
