   0:	55                   	push   %rbp
   1:	48 89 e5             	mov    %rsp,%rbp
   4:	c7 45 f8 00 00 00 00 	movl   $0x0,-0x8(%rbp)
   b:	c7 45 fc 00 00 00 00 	movl   $0x0,-0x4(%rbp)
  12:	eb 0a                	jmp    1e <main+0x1e>
  14:	8b 45 fc             	mov    -0x4(%rbp),%eax
  17:	01 45 f8             	add    %eax,-0x8(%rbp)
  1a:	83 45 fc 01          	addl   $0x1,-0x4(%rbp)
  1e:	83 7d f8 09          	cmpl   $0x9,-0x8(%rbp)
  22:	7e f0                	jle    14 <main+0x14>
  24:	8b 45 f8             	mov    -0x8(%rbp),%eax
  27:	c9                   	leaveq 
  28:	c3                   	retq   
