import time 
# NOTE I think I don't need to define self.wandb, just import wandb
class Logger:
    def __init__(self, wandb):
        self.wandb = wandb

    def log_step(self, step, epoch, train_loss, train_acc, ts_total, ts_step=None):
        log = {
            'Train Loss': train_loss,
            'Train Acc': train_acc,
            'Iteration': step,
            'Epoch': epoch,
            'Total time': time.time() - ts_total,
        }
        if ts_step:
            log['Time/step'] = time.time() - ts_step,
        if self.wandb: self.wandb.log(log)

    def log_time_step(self, ts_step):
        log = {
            'Time/step': time.time() - ts_step,
        }
        if self.wandb: self.wandb.log(log)

    def log_train_IN(self, step, epoch, train_loss, time_batch, ts_start):
        log = {
            'Train Loss': train_loss,
            'Iteration': step,
            'Epoch': epoch,
            'Total time': time.time() - ts_start,
            'Time/step': time_batch,
        }
        if self.wandb: self.wandb.log(log) 

    def log_eval(self, step, epoch, acc, test_loss, ts_eval, ts_steps_eval, validation=True):
        log = {
            'Iteration': step,
            'Epoch': epoch,
            'Time/eval': time.time() - ts_eval,
            'Time since last eval': time.time() - ts_steps_eval
        }
        if validation:
            log = {
                **log,
                'Val Accuracy': acc,
                'Val Loss': test_loss,
            }
        else:
            log = {
                **log,
                'Test Accuracy': acc,
                'Test Accuracy [avg model]': acc/100,   # NOTE /100 to make it consistent with value in log_eval_per_node()
                'Test Loss': test_loss,
            }
        if self.wandb: self.wandb.log(log)

    def log_eval_IN(self, step, epoch, acc1, acc5, test_loss, eval_time):
        log = {
            'Iteration': step,
            'Epoch': epoch,
            'Top-1 Accuracy': acc1,
            'Top-5 Accuracy': acc5,
            'Test Loss': test_loss,
            'Time/eval': eval_time,
        }
        if self.wandb: self.wandb.log(log)

    def log_eval_random_node(self, step, epoch, acc, test_loss):
        log = {
            'Iteration': step,
            'Epoch': epoch,
            'Test Accuracy [random node]': acc,
            'Test Loss [random node]': test_loss,
        }
        if self.wandb: self.wandb.log(log)

    def log_eval_per_node(self, step, epoch, acc, test_loss, acc_nodes, loss_nodes, acc_avg, loss_avg, ts_eval, ts_steps_eval):
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

    def log_quantity(self, step, epoch, x, name):
        log = {
            'Iteration': step,
            'Epoch': epoch,
            name: x,
        }
        if self.wandb: self.wandb.log(log)

    def log_weight_distance(self, step, epoch, weight_dist):
        log = {
            'Iteration': step,
            'Epoch': epoch,
            'Weight distance to init': weight_dist,
        }
        if self.wandb: self.wandb.log(log)

    def log_weight_norm(self, step, epoch, weight_norm):
        log = {
            'Iteration': step,
            'Epoch': epoch,
            'Weight L2 norm': weight_norm,
        }
        if self.wandb: self.wandb.log(log)

    def log_consensus(self, step, epoch, L2_dist):
        log = {
            'Iteration': step,
            'Epoch': epoch,
            'Consensus [L2 dist]': L2_dist,
        }
        if self.wandb: self.wandb.log(log)

    def log_grad_norm(self, step, epoch, grad_norm):
        log = {
            'Iteration': step,
            'Epoch': epoch,
            'Gradient norm': grad_norm,
        }
        if self.wandb: self.wandb.log(log)
    

    def log_weight_distance_layer0(self, step, epoch, wd_l0):
        log = {
            'Iteration': step,
            'Epoch': epoch,
            'Weight distance Layer 0': wd_l0,
        }
        if self.wandb: self.wandb.log(log)

    def log_acc(self, step, epoch, acc, loss=None, name='placeholder', validation=True):

        log = {
            'Iteration': step,
            'Epoch': epoch,
        }
        if validation:
            log[name + ' Val Accuracy'] = acc
            if loss is not None:
                log[name + ' Val loss'] =  loss
        else:
            log[name + ' Test Accuracy'] = acc
            if loss is not None:
                log[name + ' Test loss'] =  loss
                
        if self.wandb: self.wandb.log(log)

    def log_acc_IN(self, step, epoch, acc1, acc5, name='placeholder'):
        log = {
            'Iteration': step,
            'Epoch': epoch,
            name + ' Top-1 Accuracy': acc1,
            name + ' Top-5 Accuracy': acc5,
        }
        if self.wandb: self.wandb.log(log)

    def log_single_acc(self, acc, log_as='placeholder'):
        if acc < 1:
            acc *= 100
        log = {
            log_as: acc,
        }
        if self.wandb: self.wandb.log(log)

    def log_max_ema_acc(self, max_acc):
        if max_acc < 1:
            max_acc *= 100
        log = {
            'Max EMA Accuracy': max_acc,
        }
        if self.wandb: self.wandb.log(log)