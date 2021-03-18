 plot  "./histo" u 1:(-log($2))*0.02585199101165164 w lp
 replot  "./histo" u 1:($1)*($1) w lp
