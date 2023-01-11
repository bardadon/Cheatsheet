select 
    empno,
    sal,
    sum(empno) over (PARTITION BY deptno)
from emp


