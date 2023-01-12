create view V4 (id,amt,trx)
as
select 1, 100, 'PR' from t1 union all
select 2, 100, 'PR' from t1 union all
select 3, 50, 'PY' from t1 union all
select 4, 100, 'PR' from t1 union all
select 5, 200, 'PY' from t1 union all
select 6, 50, 'PY' from t1;


select * from V4;

/*
"id","amt","trx"
1,100,"PR"
2,100,"PR"
3,50,"PY"
4,100,"PR"
5,200,"PY"
6,50,"PY"
*/

-- Solution:

