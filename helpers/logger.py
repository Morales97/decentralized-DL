import time 

class Logger:
    def __init__(self, wandb):
        self.wandb = wandb
        self.accuracies = []
        self.accuracies_per_node = []
        self.test_losses = []
        self.test_losses_per_node = []
        self.train_losses = []
        self.weight_distance = []

    def log_step(self, step, epoch, train_loss, ts_total, ts_step):
        self.train_losses.append(train_loss)
        log = {
            'Train Loss': train_loss,
            'Iteration': step,
            'Epoch': epoch,
            'Total time': time.time() - ts_total,
            'Time/step': time.time() - ts_step,
        }
        if self.wandb: self.wandb.log(log)

    def log_eval(self, step, epoch, acc, test_loss, ts_eval, ts_steps_eval):
        self.accuracies.append(acc)
        self.test_losses.append(test_loss)
        log = {
            'Iteration': step,
            'Epoch': epoch,
            'Test Accuracy': acc,
            'Test Loss': test_loss,
            'Time/eval': time.time() - ts_eval,
            'Time since last eval': time.time() - ts_steps_eval
        }
        if self.wandb: self.wandb.log(log)

    def log_eval_random_node(self, step, epoch, acc, test_loss):
        self.accuracies.append(acc)
        self.test_losses.append(test_loss)
        log = {
            'Iteration': step,
            'Epoch': epoch,
            'Test Accuracy [random node]': acc,
            'Test Loss [random node]': test_loss,
        }
        if self.wandb: self.wandb.log(log)

    def log_eval_per_node(self, step, epoch, acc, test_loss, acc_nodes, loss_nodes, acc_avg, loss_avg, ts_eval, ts_steps_eval):
        self.accuracies.append(acc)
        self.test_losses.append(test_loss)
        log = {
            'Iteration': step,
            'Epoch': epoch,
            'Test Accuracy': acc,
            'Test Loss': test_loss,
            'Test Accuracy [min]': min(acc_nodes),
            'Test Accuracy [max]': max(acc_nodes),
            'Test Loss [min]': min(loss_nodes),
            'Test Loss [max]': max(loss_nodes),
            'Test Accuracy [avg model]': acc_avg,
            'Test Loss [avg model]': loss_avg,
            'Time/eval': time.time() - ts_eval,
            'Time since last eval': time.time() - ts_steps_eval
        }
        if self.wandb: self.wandb.log(log)

    def log_weight_distance(self, step, epoch, weight_dist):
        self.weight_distance.append(weight_dist)
        log = {
            'Iteration': step,
            'Epoch': epoch,
            'Weight distance to init': weight_dist,
        }
        if self.wandb: self.wandb.log(log)
