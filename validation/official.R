args<-commandArgs(TRUE); root<-normalizePath(args[1],winslash='/')
.libPaths(c(file.path(root,'Rlib'),.libPaths()))
library(fixest);library(data.table);library(Matrix)
setFixest_nthreads(1)
setFixest_estimation(fixef.tol=3e-12)
source(file.path(root,'external/did2s/R/did2s.R'))
meta<-read.csv(file.path(root,'python_results.csv'));out<-list()
for(i in seq_len(nrow(meta))){
 m<-meta[i,];d<-read.csv(file.path(root,sprintf('fixtures/fixture_%02d.csv',m$fixture)))
 first<-if(m$covariates) ~x|state+year else ~0|state+year
 start<-proc.time()[3]
 fit<-did2s(d,'y',first,~D,'D','state',weights=if(m$weighted) 'w' else NULL)
 result<-coeftable(fit)
 out[[i]]<-data.frame(fixture=m$fixture,estimate=result[1,1],se=result[1,2],stat=result[1,3],p=result[1,4],df_t_inferred=nrow(d)-1,p_t_check=2*pt(-abs(result[1,3]),nrow(d)-1),seconds=unname(proc.time()[3]-start))
}
write.csv(do.call(rbind,out),file.path(root,'r_results.csv'),row.names=FALSE)
writeLines(c(capture.output(sessionInfo()),paste('did2s source commit:',system(paste('git -C',shQuote(file.path(root,'external/did2s')),'rev-parse HEAD'),intern=TRUE))),file.path(root,'r_session.txt'))
