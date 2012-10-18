/* unr1 */
for (int i = 0; i < 1000; ++i)
  do_something(i);
/* end */
/* unr2 */
for (int i = 0; i < 500; i+=2)
{
  do_something(i);
  do_something(i+1);
}
/* end */
/* spl1 */
for (int i = 0; i < 1000; ++i)
{
  do_a(i);
  do_b(i);
}
/* end */
/* spl2 */
for (int i = 0; i < 500; i+=2)
{
  do_a(i);
  do_a(i+1);
  do_b(i);
  do_b(i+1);
}
/* end */
