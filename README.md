# population-resources

Code to simulate population (P) and depleting resources (R) coupled dynamical systems model:

```math
	\begin{align}
		 \dot{P} & =  P(\alpha - \beta P/R)\, , \\
		 \dot{R} &= \delta -\gamma P  \, .
	\end{align}
```

for $\alpha, \beta, \gamma, \delta >0$ and $R_0 >> P_0 >0 .$

See <a href="https://github.com/saf92/population-resources/blob/main/pop_res.pdf">
         paper </a> for details and application to world human population. 

![alt text](https://github.com/saf92/population-resources/blob/main/plots/pop_res_sim.png)

