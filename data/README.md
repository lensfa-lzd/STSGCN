# About Dataset

The adj file in Metr-La dataset is a distance base graph build by using thresholded 
Gaussian kernel (Shuman et al., 2013)

$$
{{\rm{W}}_{ij}} = \exp ( - \frac{{dist{{({v_i},{v_j})}^2}}}{{{\sigma ^2}}})
$$

and if dist > threshold, the value is set to 0,

Threshold is assigned to 0.1 in this case.

Base on this distance base graph, we built a Top-K neighbor graph mask by top k% value to 1, 
otherwise to 0, and after further process making the attention only focus on the top k% node.

In the experiment, we actually built a group Top-K graph mask by setting,

Top 10% to 0,

Top 10%-20% to 1, 

...

Top 90%-100% to 9.

