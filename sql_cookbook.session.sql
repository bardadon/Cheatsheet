-- Method #1:
select 
sal,
row_number() over(PARTITION BY sal) as occurances
from emp
where deptno = 20
ORDER BY occurances desc
limit 1;

/*
"sal","occurances"
800,"2"
*/




