BSXYvsini2017New.m: esse código gera as figuras referentes à técnica boorstrap presente na Seção 4.2.1 do paper;

MapsXY2020VmagCondition.m: esse código faz uma análise do impacto da magnitude na distribuição de vsini. Ele tenta resolver a seguinte questão: se essa distribuição por área for organizada não pelo vsini, mas pela Vmag, o Ring permanece? Os primeiros resultados apontam que depende do tipo de quartil.

bootrsp.m: esse é o código para gerar as amostras bootstrap.

Vão aparecer três janelas que correspondem, respectivamente, o tamanho do lado do quadrado (isto é, se a área for 20X20, digite 20), o tamanho da amostra bootstrap (use 1000) e, por último, digite 150 para cobrir uma área de lado 300 pc.

Veja como você pode progredir com tais códigos em conjunto com os novos dados do CGS. 

Nós também poderiamos pensar em fazer o seguinte procedimento: 
Calcular o valor médio de vsini em cada área e comparar com a distribuição bootstrap. Assim, poderiamos verificar o quanto o valor médio (ou mesmo a mediana, já que é mais confiável que a média) se distancia do valor central da média bootstrap. Com esse procedimento, podemos medir a distância em sigma entre esses valores de vsini (bootstrap e média empírica). Se os valores empíricos caem dentro de 1sigma, nos resultados são mais confiáveis, caso contrário, devemos excluir "essa área" dos nossos resultados.
