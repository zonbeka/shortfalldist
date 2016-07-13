clear all
%THESIS work%

%% PH distribution parameters for demand distribution
d=[0.55 0.45];     %Initial probability vector for demand distribution
Td=[0.3 0.3;
    0.15 0.2];    %Transition probability matrix of demand distn
td=[0.4;
    0.65];        %Absorption probability vector
dq1=0;
expdemand=0;
for x=1:9
    dq(x)= d*Td^(x-1)*td;  %PDF of demand distribution
    dq1=dq1+dq(x);
    expdemand=expdemand+(x*dq(x));
end

sumdd = 0;
for dd = 1:9
    sumdd = sumdd + dq(dd);
end


%% PH distribution parameters for single unit service time distribution
s=[0.6 0.4];       % Initial probability vector
Ts=[0.15 0.2;
    0.05 0.1];      %Transition matrix assuming 2 phases
ts=[0.65;
    0.85];          % Absorption matrix
M1=0;
for x=1:8
    M(x)= s*Ts^(x-1)*ts; %PDF of service distribution
    M1=M1+M(x);
end
M(9)=1-M1;
Msum=sum(M,2);
expserv=0;
for x=1:9
    expserv=expserv+(x*M(x));
end

sumss = 0;
for ss = 1:9
    sumss = sumss + M(ss);
end

%% PH distribution for service time of batch
b=kron(d,s);     %Initial Probability vector which is Kronecker product of d & m according to Theorem 2.6.3 Latouche & Ramswami
Id=[1 0;
    0 1];        % Identity matrix of size equal to demand transition matrix

Tb=kron(Id,Ts)+kron(Td,ts*s);  %Transition matrix of service distribution having random demand and service time

l=[1 1 1 1]';
tb=l-(Tb*l);                   % Absorption probabilities
servexpect=0;
for x=1:80
    k22(x)=(b*Tb^(x-1))*tb;           %PDF of service time distribution
    servexpect=servexpect+(x*k22(x)); % service time expectation
end

sumk22 = 0;
for kk = 1:9
    sumk22 = sumk22 + k22(kk);
end

%% Markov chain for two suppliers assuming that orders are placed in every
%% period and service time unit is same as period unit

A0=tb*b;                   %Probability matrix for situation when services at both the suppliers finish in next period starting from (1 0)

A1=Tb;             %Prob matrix : starting from (1 0) service at S1 continues and in next period order is placed to S2
A2=kron(b,Tb);
A3=kron(Tb,b);          %Probability matrix for situation when services at both the suppliers finish in next period starting not from (1 0)
A4=kron(l,b);
A5=kron(kron(tb,tb),b);            % Service at both suppliers continues
A6=kron(Tb,Tb);          %Serice at S1 finishes and S2 continues
A7=kron(Tb,tb);          % Service at S1 continues while at S2 finishes
A8=kron(tb,Tb);
A9=kron(Tb,kron(tb,b));
A10=kron(kron(tb,b),Tb);
A11=kron(kron(l,tb),b);           % Service at 2 finishes and services at S1 too finishes n new order goes to S1.
A12=kron(kron(l,b),Tb);
A13=kron(kron(tb,l),b);
A14=kron(Tb,kron(l,b));             %At boundry state service at supplier S1 finishes and service at S2 continues
A15=kron(kron(l,tb),kron(b,b));             %At boundry state S2 finishes and S1 continues
A16=kron(kron(tb,l),kron(b,b));



U=zeros(4, 16);  
U2=zeros(16,16);
U3=zeros(4,4);
U4=zeros(16,4);

                                                            % 4X16 matrix of zeros

F=[A0 A1 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U;
   A0 U3 A1 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U;
   A0 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 A2 U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U;
   A0 U3 U3 U3 A1 U3 U3 U3 U3 U3 U3 U3 U3 U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U;
   A0 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U  A2 U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U;
   A0 U3 U3 U3 U3 U3 A1 U3 U3 U3 U3 U3 U3 U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U;
   A4 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U;
   A0 U3 U3 U3 U3 U3 U3 U3 A1 U3 U3 U3 U3 U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U;
   A0 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U  U  A3 U  U  U  U  U  U  U  U  U  U  U  U  U  U  U;
   A0 U3 U3 U3 U3 U3 U3 U3 U3 U3 A1 U3 U3 U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U;
   A0 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U  U  U  A3 U  U  U  U  U  U  U  U  U  U  U  U  U  U;
   A0 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 A1 U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U;
   A4 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U3 U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U  U;
   A5 U4 U4 A8 U4 U4 U4 A7 U4 U4 U4 U4 U4 U2 U2 U2 U2 A6 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2;
   A5 U4 U4 U4 U4 A8 U4 A7 U4 U4 U4 U4 U4 U2 U2 U2 U2 U2 A6 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2;
   A5 A8 U4 U4 U4 U4 U4 U4 U4 A7 U4 U4 U4 U2 U2 U2 U2 U2 U2 A6 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2;
   A5 A8 U4 U4 U4 U4 U4 U4 U4 U4 U4 A7 U4 U2 U2 U2 U2 U2 U2 U2 A6 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2;
   A5 U4 U4 U4 A8 U4 U4 U4 A7 U4 U4 U4 U4 U2 U2 U2 U2 U2 U2 U2 U2 A6 U2 U2 U2 U2 U2 U2 U2 U2 U2;
   A5 U4 U4 U4 U4 U4 A8 U4 A7 U4 U4 U4 U4 U2 U2 U2 U2 U2 U2 U2 U2 U2 A6 U2 U2 U2 U2 U2 U2 U2 U2;
   A5 U4 A8 U4 U4 U4 U4 U4 U4 U4 A7 U4 U4 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 A6 U2 U2 U2 U2 U2 U2 U2;
   A5 U4 A8 U4 U4 U4 U4 U4 U4 U4 U4 U4 A7 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 A6 U2 U2 U2 U2 U2 U2;
   A5 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U2 A10 A9 U2 U2 U2 U2 U2 U2 U2 U2 U2 A6 U2 U2 U2 U2 U2;
  A13 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U2 U2 A14 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2;
   A5 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 A10 U2 U2 A9 U2 U2 U2 U2 U2 U2 U2 U2 U2 A6 U2 U2 U2 U2;
  A11 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 A12 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2; 
   U4 A5 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U2 U2 U2 U2 U2 A10 A9 U2 U2 U2 U2 U2 U2 U2 A6 U2 U2 U2;
   U4 A5 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U2 U2 U2 U2 A10 U2 U2 A9 U2 U2 U2 U2 U2 U2 U2 A6 U2 U2;
   U4 U4 A5 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U2 U2 U2 U2 U2 U2 U2 U2 U2 A10 A9 U2 U2 U2 U2 U2 A6 U2;
   U4 U4 A5 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U2 U2 U2 U2 U2 U2 U2 U2 A10 U2 U2 A9 U2 U2 U2 U2 U2 A6;
   U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 A16 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 A14 U2 U2 U2 U2;
   U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 U4 A15 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 U2 A12 U2 U2 U2 U2 U2];
   
prob=F^500;
index=[1010;1020;1030;1040; 2010;2020;2030;2040; 3010;3020;3030;3040; 5010;5020;5030;5040;
       6010;6020;6030;6040; 8010;8020;8030;8040; 9010;9020;9030;9040; 0201;0202;0203;0204;
       0301;0302;0303;0304; 0501;0502;0503;0504; 0601;0602;0603;0604; 0801;0802;0803;0804;
       0901;0902;0903;0904; 
       4111;4121;4131;4141; 4112;4122;4132;4142; 4113;4123;4133;4143; 4114;4124;4134;4144;
       7111;7121;7131;7141; 7112;7122;7132;7142; 7113;7123;7133;7143; 7114;7124;7134;7144;
       1411;1421;1431;1441; 1412;1422;1432;1442; 1413;1423;1433;1443; 1414;1424;1434;1444;
       1711;1721;1731;1741; 1712;1722;1732;1742; 1713;1723;1733;1743; 1714;1724;1734;1744;
       5211;5221;5231;5241; 5212;5222;5232;5242; 5213;5223;5233;5243; 5214;5224;5234;5244;
       8211;8221;8231;8241; 8212;8222;8232;8242; 8213;8223;8233;8243; 8214;8224;8234;8244;
       2511;2521;2531;2541; 2512;2522;2532;2542; 2513;2523;2533;2543; 2514;2524;2534;2544;
       2811;2821;2831;2841; 2812;2822;2832;2842; 2813;2823;2833;2843; 2814;2824;2834;2844;
       6311;6321;6331;6341; 6312;6322;6332;6342; 6313;6323;6333;6343; 6314;6324;6334;6344;
       9311;9321;9331;9341; 9312;9322;9332;9342; 9313;9323;9333;9343; 9314;9324;9334;9344;
       3611;3621;3631;3641; 3612;3622;3632;3642; 3613;3623;3633;3643; 3614;3624;3634;3644;
       3911;3921;3931;3941; 3912;3922;3932;3942; 3913;3923;3933;3943; 3914;3924;3934;3944;
       7411;7421;7431;7441; 7412;7422;7432;7442; 7413;7423;7433;7443; 7414;7424;7434;7444;
       4711;4721;4731;4741; 4712;4722;4732;4742; 4713;4723;4733;4743; 4714;4724;4734;4744;
       8511;8521;8531;8541; 8512;8522;8532;8542; 8513;8523;8533;8543; 8514;8524;8534;8544;
       5811;5821;5831;5841; 5812;5822;5832;5842; 5813;5823;5833;5843; 5814;5824;5834;5844;
       9611;9621;9631;9641; 9612;9622;9632;9642; 9613;9623;9633;9643; 9614;9624;9634;9644;
       6911;6921;6931;6941; 6912;6922;6932;6942; 6913;6923;6933;6943; 6914;6924;6934;6944];
    

time = 3;
age = 9;
TransitionM = zeros(10000,10000);
for rr = 1:size(index) 
    for cc = 1:size(index)
        TransitionM(index(rr),index(cc)) = F(rr,cc);
    end
end

    
Pim=zeros(10000,1);
for i=1:size(index)
    Pim(index(i),1)=prob(1,i);
end

WW = zeros(3,1);
%%States Check
ST = zeros(100,1);
for i = 0:age
    if i == 0
        j = 1:age;
        for n = 1:4
            ST(10*i+j) = ST(10*i+j) + Pim(100*j+n);
        end
    else
        for j = 0:age
            if j == 0
                for m = 1:4
                    ST(10*i+j) = ST(10*i+j) + Pim(1000*i+10*m);
                end
            
            else
                for m = 1:4
                    for n=1:4
                        ST(10*i+j) = ST(10*i+j) + Pim(1000*i+100*j+10*m+n);
                    end
                end
            end
        end
    end
end
%% PPRR is the matrix of transition probability between two states.
PPRR = zeros(99,99);
for aa = 2:99
    for bb = 2:99
        for x = 0:4
            for y = 0:4
                for p = 0:4
                    for q = 0:4
                        PPRR(aa,bb) = PPRR(aa,bb) + Pim(aa*100+10*x+y) * TransitionM(aa*100+10*x+y,bb*100+10*p+q);
                    end
                end
            end
        end
    end
end
%% Leadtime Distribution
P = zeros(9,1);
R = zeros(9,1);
for i = 0:9
    if i == 0
        for j = 1:9
            LL = j;
            R(j) = R(j) + PPRR(LL,10);
        end
    else
        for j = 0:9
            if j == 0
                LL = 10*i;
                P(i) = P(i) + PPRR(LL,10);
            else
                KK = 10*i+j;
                for qq = 1:KK
                    P(i) = P(i) + PPRR(KK,qq);
                end
                for m1 = 1:9
                    for m2 = 0:j
                        R(j)=R(j)+ PPRR(KK,m1*10+m2);
                    end
                end
            end
        end
    end
end

LT = zeros(9,1);
for i = 1:9
    LT(i)=(P(i)+R(i))/(sum(P)+sum(R));
end
    
expectedLT = 0;
for i = 1:9
    expectedLT = expectedLT + i*LT(i);
end

LTP = zeros(3,1);
LTP(1) = sum(LT(1)+LT(2)+LT(3));
LTP(2) = sum(LT(4)+LT(5)+LT(6));
LTP(3) = sum(LT(7)+LT(8)+LT(9));

expectedLTP=0;
for i = 1:3
    expectedLTP = expectedLTP+i*LTP(i);
end
    
    

%% Wait Time distribution
PR = zeros(99,3);
PART00 = PPRR(10,10)+PPRR(20,10)+PPRR(50,10)+PPRR(80,10)+PPRR(2,10)+PPRR(5,10)+PPRR(8,10)+PPRR(14,10)+PPRR(41,10)+PPRR(52,10)+PPRR(25,10)+PPRR(17,10)+PPRR(71,10)+PPRR(82,10)+PPRR(28,10);
ArrivalST = ST(30)+ST(03)+ST(60)+ST(06)+ST(90)+ST(09)+ST(36)+ST(63)+ST(39)+ST(93)+ST(69)+ST(96)+PART00;
% Wait Time == 1
PR(36,1) = PPRR(36,47)*(sum(PPRR(47,:))-PPRR(47,58))/ST(47)/ArrivalST;
PR(63,1) = PPRR(63,74)*(sum(PPRR(74,:))-PPRR(74,85))/ST(74)/ArrivalST;
PR(69,1) = PPRR(69,74)*(sum(PPRR(74,:))-PPRR(74,85))/ST(74)/ArrivalST;
PR(96,1) = PPRR(96,47)*(sum(PPRR(47,:))-PPRR(47,58))/ST(47)/ArrivalST;

% Wait Time == 2
PR(36,2) = PPRR(36,47)*PPRR(47,58)/ST(47)*(sum(PPRR(58,:))-PPRR(58,69))/ST(58)/ArrivalST;
PR(63,2) = PPRR(63,74)*PPRR(74,85)/ST(74)*(sum(PPRR(85,:))-PPRR(85,96))/ST(85)/ArrivalST;
PR(69,2) = PPRR(69,74)*PPRR(74,85)/ST(74)*(sum(PPRR(85,:))-PPRR(85,96))/ST(85)/ArrivalST;
PR(96,2) = PPRR(96,47)*PPRR(47,58)/ST(47)*(sum(PPRR(58,:))-PPRR(58,69))/ST(58)/ArrivalST;

% Wait Time == 3
PR(36,3) = PPRR(36,47)*PPRR(47,58)/ST(47)*PPRR(58,69)/ST(58)/ArrivalST;
PR(63,3) = PPRR(63,74)*PPRR(74,85)/ST(74)*PPRR(85,96)/ST(85)/ArrivalST;
PR(69,3) = PPRR(69,74)*PPRR(74,85)/ST(74)*PPRR(85,96)/ST(85)/ArrivalST;
PR(96,3) = PPRR(96,47)*PPRR(47,58)/ST(47)*PPRR(58,69)/ST(58)/ArrivalST;

WT = zeros(1,4);
WT(2) = sum(PR(:,1));
WT(3) = sum(PR(:,2));
WT(4) = sum(PR(:,3));
WT(1) = 1-WT(1)-WT(2)-WT(3);
expectedWT = 1*WT(2)+2*WT(3)+3*WT(4);

%% Joint Distribution of Quantity and Service Time
qt1 = zeros(9,1);
for servtime = 1:9
    qt1(servtime,1) = M(servtime);
end

Qt = zeros(90,9);
QT = zeros(9,9);
qtt = qt1;
for orderqty = 1:9
    if orderqty == 1
        qtt = qt1;
    else
        qtt = conv(qtt,qt1);
    end
    Qt(orderqty:(orderqty+(size(qtt)-1)),orderqty) = qtt;    
   
end

for orderqty = 1:9
    for servtime = 1:9
        QT(1:8,orderqty) = Qt(1:8,orderqty);
        QT(9,orderqty) = sum(Qt(9:size(Qt,1),orderqty));
    end
end
%% JointQT 
JointQT = zeros(9,9);
for orderqty = 1:9
    JointQT(:,orderqty) = dq(orderqty)* QT(:,orderqty);
end


%% Grounded Pieces with Wait Time
pcase = zeros(100000000,1);
sumq = 0;
for so = 1:9
    for qo = 1:9
        for sm =1:9
            for qm = 1:9
                for sn = 1:9
                    for qn = 1:9
                        if so == 1 
                            if sm == 1
                                sumq = qn;
                                pcase(sumq*10^6+so*10^5+qo*10^4+sm*1000+qm*100+sn*10+qn)=JointQT(so,qo)*JointQT(sm,qm)*JointQT(sn,qn);
                            elseif sm >= 2 && sm <= 3 
                                sumq = qm+qn;
                                for wm = 1:sm
                                    pcase(sumq*10^6+so*10^5+qo*10^4+sm*1000+qm*100+sn*10+qn)=pcase(sumq*10^6+so*10^5+qo*10^4+sm*1000+qm*100+sn*10+qn)+JointQT(so,qo)*(WT(wm)*JointQT(sm-(wm-1),qm))*JointQT(sn,qn);
                                end
                            else 
                                sumq = qm+qn;
                                for wm = 1:4
                                    pcase(sumq*10^6+so*10^5+qo*10^4+sm*1000+qm*100+sn*10+qn)=pcase(sumq*10^6+so*10^5+qo*10^4+sm*1000+qm*100+sn*10+qn)+JointQT(so,qo)*(WT(wm)*JointQT(sm-(wm-1),qm))*JointQT(sn,qn);
                                end                               
                            end
                        elseif so >=2 && so <= 6
                            if sm == 1
                                sumq = qn;
                                for wo = 1:min(so,4)
                                    pcase(sumq*10^6+so*10^5+qo*10^4+sm*1000+qm*100+sn*10+qn)=pcase(sumq*10^6+so*10^5+qo*10^4+sm*1000+qm*100+sn*10+qn)+(WT(wo)*JointQT(so-(wo-1),qo))*JointQT(sm,qm)*JointQT(sn,qn);
                                end
                            elseif sm >= 2 && sm <= 3
                                sumq = qn;
                                for wo = 1:min(so,4)
                                    for wm = 1:sm
                                        pcase(sumq*10^6+so*10^5+qo*10^4+sm*1000+qm*100+sn*10+qn)=pcase(sumq*10^6+so*10^5+qo*10^4+sm*1000+qm*100+sn*10+qn)+(WT(wo)*JointQT(so-(wo-1),qo))*(WT(wm)*JointQT(sm-(wm-1),qm))*JointQT(sn,qn);
                                    end
                                end
                            else
                               sumq = qm+qn;
                               for wo = 1:min(so,4)
                                    for wm = 1:4
                                        pcase(sumq*10^6+so*10^5+qo*10^4+sm*1000+qm*100+sn*10+qn)=pcase(sumq*10^6+so*10^5+qo*10^4+sm*1000+qm*100+sn*10+qn)+(WT(wo)*JointQT(so-(wo-1),qo))*(WT(wm)*JointQT(sm-(wm-1),qm))*JointQT(sn,qn);
                                    end
                               end
                            end
                        
                        elseif  so >= 7
                            if  sm == 1
                                sumq = qo+qn;
                                 for wo = 1:4                                   
                                        pcase(sumq*10^6+so*10^5+qo*10^4+sm*1000+qm*100+sn*10+qn)=pcase(sumq*10^6+so*10^5+qo*10^4+sm*1000+qm*100+sn*10+qn)+(WT(wo)*JointQT(so-(wo-1),qo))*JointQT(sm,qm)*JointQT(sn,qn);                                    
                                 end
                            elseif sm >= 2 && sm <= 3
                                sumq = qo+qn;
                                for wo = 1:4
                                    for wm = 1:sm
                                        pcase(sumq*10^6+so*10^5+qo*10^4+sm*1000+qm*100+sn*10+qn)=pcase(sumq*10^6+so*10^5+qo*10^4+sm*1000+qm*100+sn*10+qn)+(WT(wo)*JointQT(so-(wo-1),qo))*(WT(wm)*JointQT(sm-(wm-1),qm))*JointQT(sn,qn);
                                    end
                                end                                
                            else
                                sumq = qo+qm+qn;
                                for wo = 1:4
                                    for wm = 1:4
                                        pcase(sumq*10^6+so*10^5+qo*10^4+sm*1000+qm*100+sn*10+qn)=pcase(sumq*10^6+so*10^5+qo*10^4+sm*1000+qm*100+sn*10+qn)+(WT(wo)*JointQT(so-(wo-1),qo))*(WT(wm)*JointQT(sm-(wm-1),qm))*JointQT(sn,qn);
                                    end
                                end                                 
                            end
                        end
                    end
                end
            end
        end
    end
end                        


sf = zeros(30,1);
expsf = zeros(30,1);
for i = 1:30
    sf(i) = sum(pcase(i*10^6:((i+1)*10^6-1)));
    expsf(i) = i*sf(i);
end
                            
sumexpsf = sum(expsf);
                        
%% Base Stock Level
fillrate=0.95;
aana=(1-fillrate)*expdemand;
zz = zeros(30,1);

for basestock=1:30
    zz(basestock)=0;
    for shortfall=1:27
    zz(basestock)=zz(basestock)+sf(shortfall)*max(shortfall-basestock,0);
    end
end
%% Lowest Base Stock Level
fprintf('Minimum Base Stock = %2.0f\n',min(find(zz(:,1) <= aana)));

%% Safety Stock
safety = min(find(zz(:,1) <= aana))-(expectedLTP+1)*expdemand;
fprintf('Minimum Safety Stock = %2.5f\n',safety);

subplot(2,2,1),bar(1:9,LT(1:9,1));
set(gca,'XTick',[2:2:10]);
title('Leadtime Distribution in time slot')
xlabel('Time slots')
ylabel('Probability')
subplot(2,2,2),bar(1:3,LTP(1:3,1));
set(gca,'XTick',[1:3]);
set(gca,'Ytick',[0:0.2:1]);
title('Leadtime Distribution in period')
xlabel('Time periods')
ylabel('Probability')
subplot(2,2,3),bar(1:3,WT(1,2:4));
set(gca,'XTick',[1:3]);
set(gca,'Ytick',[0:0.003:0.01]);
set(gca,'YTickLabel',str2mat('   0','0.003','0.006','0.009'))
title('Waiting time distribution')
xlabel('Time periods')
ylabel('Probability')
subplot(2,2,4), bar(1:27,sf(1:27,1));
set(gca,'XTick',[0:5:30]);
title('Shortfall Distribution')
xlabel('Units')
ylabel('Probability')
