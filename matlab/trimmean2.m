% This function takes a data vector "x", and a trimming % "p" and computes
% the trimmed mean by ignoring the top & bottom p/2 percent of the data.

function out = trimmean2(x,p)
  n = length(x);
  perc = 100*((1:n)-.5)/n;
  x = sort(x);
  newx = x((perc>=(p/2))&(perc<=100-p/2));
  out = mean(newx);
