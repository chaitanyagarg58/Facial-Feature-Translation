import tensorflow as tf
from tensorboardX import SummaryWriter
import warnings

tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        # self.writer.add_summary(summary, step)
        
        # Create a summary writer
        writer = tf.summary.create_file_writer("log_dir")  # Replace "log_dir" with your desired log directory

        # Create a summary
        with writer.as_default():
            tf.summary.scalar(tag, value, step=step)

        # Make sure to close the writer when you're done
        writer.close()