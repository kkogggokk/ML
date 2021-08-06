import argparse 

parser = argparse.ArgumentParser()

#parser.add_argument(<짧은플래스>,<긴플래스>)
parser.add_argument('-l','--left',type=int) # 타입설정
parser.add_argument('-r','--right',type=int)
parser.add_argument(
    '--operation',
    dest = 'op',# 타겟속성, 기본은 -- 없이
    help = 'Set Operation', # 인자 설명
    default = 'sum' # 기본값 
)

args = parser.parse_args()
print(args)

if args.op == 'sum':
    out = args.left + args.right
elif args.op =='sub':
    out = args.left - args.right
    
print(out)