


# TODO implement membership inference attack of Yeom et al
# run a fwd pass on final model on train set -> get expected train loss per sample
# define a 20k set (10k random train + 10k test) to evaluate DP on
# can also introduce STD of train loss and use it to refine DP prediction
# more elaborate membership attacks: Shkori et al (multiple shadow models) and Salem et al (one shadow model)
# Lastly, Tramer et al (2022) propose a superior membership attack