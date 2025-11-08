from Optz.optimizer import HubbardOptimizer

if __name__ == "__main__":

    optimizer = HubbardOptimizer(
        compound="MnO",
        target_gap=4.0,
        initial_u=[0.0001], 
        bounds=(0.0001, 10),
        max_iter=30,
        precision=0.01
    )
    optimizer.bayesian_optimization()
