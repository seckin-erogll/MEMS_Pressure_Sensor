clc
close all


color1 = [0 0.4470 0.7410];
color2 = [0.8500 0.3250 0.0980];
color3 = [0.9290 0.6940 0.1250];
color4 = [0.4940 0.1840 0.5560];
color5 = [0.4660 0.6740 0.1880];
color  = {color1 color2 color3 color4 color5};

%%%%%%%%    ANALYTICAL MODEL OF PRESSURE SENSOR  %%%%%%%%
%   Important notes!
%   "0" of y- axis is determined as the top of the diaphragm
%   t_1 t_2 t_3 is numerated from the top layer of the top electrode

%% %%%%%%%%    DESIGN PARAMETERS   %%%%%%%%

dr  = 1;                             %Radial density
dt  = 1;                             %Thickness density 
a = linspace(410e-6, 410e-6, dr);    %µm, Diaphragm radius
g = 0.0000000000010e-6;                         %µm, Gap 10um
t_1 = 1.2e-6;                         %µm, Top Parylene thickness
t_2 = 0.22e-6;                        %µm, Top Electrode thickness
t_3 = linspace(3.79e-6,3.79e-6,dt);   %µm, Parylene thickness changing 2-10µm
t_4 = 0.71e-6;                        %µm, Bottom Parylene thickness

d_P = 100;                           %Pa, Pressure change density
P = 0:d_P:10000;                     %Pa, Pressure Range
%Mechanical Properties

E_p = 3.2e9;                         %GPa, Elastic Modulus of parylene 
E_au = 70e9 ;                        %GPa, Elastic Modulus of Gold
v_p = 0.33  ;                        %Poisson's ratio of parylene
v_au = 0.44 ;                        %Poisson's ratio of gold

%Electrical Properties
e_p = 3.15;                         %Dielectric constant of parylene
e_air = 1 ;                         %Dielectric constant of air
e_0 = 8.85e-12 ;                    % Vacuum permittivity Farad/meter

Sensitivitynontouch = [];
Sensitivitytouch    = [];


for i = 1:length(a)           %Radial indice

    Sensitivity1 = [];        %Sensitivity values stored for nonTouch mode
    ntzero = [];              %Initial value of the fit at cap = 0
    rsqnt = [];               %R^2 

    Sensitivity2 = [];        %Sensitivity values stored for Touch mode     
    tzero = [];               %Initial value of the fit at cap = 0  
    rsqtm = [];               %R^2

    % Create a nested structure for this radius
    %figure
    %hold on
    for j = 1:length(t_3)       %Thickness indice. For a radius value, sensitivity
                                %values are stored for all thickness values.
        
        %%% Store variables
        R = a(i);              %Corresponding Radius            
        E = [E_p E_au E_p];    %Elastic Modulus
        t = [t_1 t_2 t_3(j)];  %Corresponding Thickness 
        v = [v_p v_au v_p];    %Poisson's Ratio
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Calculation of Linear Spring Coefficient
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% CALCULATION OF NEUTRAL AXIS POSITION %%%

        %When the eqn is solved by hand, following simplification occured:
        % d = -(c_2 * t_1 + c_3*(t_1+t_2) / (0.5*(c_1 + c_2 + c_3)),
        c_1 = (E(1) .* t(1)) ./ (1 - (v(1).^2));
        c_2 = (E(2) .* t(2)) ./ (1 - (v(2).^2));
        c_3 = (E(3) .* t(3)) ./ (1 - (v(3).^2));
        
        %Thus, neutral axis d is as follows
        %d = -(c_2 * t_1 + c_3 *(t_1 + t_2)) ./ (0.5 .* (c_1 + c_2 + c_3));
        % Now, determine "b" values for linear spring coefficient
        h_tot = t(1) + t(2) + t(3);     

        %% Alican's equation unmodified 
        %
        % "2" stayed the same and it is w.r.t geometric midplane
        d = -(c_2 * t_1 + c_3 *(t_1 + t_2)) ./ (2 .* (c_1 + c_2 + c_3));
        d = abs(d)
        b_1 = (h_tot./2)-d;
        b_2 = t(3) - b_1; 
        b_4 = (h_tot./2) + d;
        b_3 = b_4 - t_1;
        b = [b_1 b_2 b_3 b_4];
        
        % While subtracting b values, abs() is not taken

        sum_1 = (E(3) .* (b(2)^3 - b(1)^3 )) ./ (1- (v(3).^2));
        sum_2 = (E(2) .* (b(3)^3 - b(2)^3 )) ./ (1- (v(2).^2));
        sum_3 = (E(1) .* (b(4)^3 - b(3)^3 )) ./ (1- (v(1).^2));
        k_1 = sum_1 + sum_2 + sum_3;
        %}
        
               
        k_lin = (64 .* pi .* k_1) ./ (R.^2) %Linear spring coefficient
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Calculation of Nonlinear spring coefficient
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % In order to calculate each flexural rigidity D=E⋅t^3/12(1−ν^2) is
        % implemented

        D = zeros(1,3);
        for fr  = 1:3
            D(fr) = (E(fr) .* (t(fr).^3)) ./ (12*(1-v(fr).^2));
        end

        %% Calculating mean Poisson's Ratio
        %Product notation is actually a constant. Therefore it is pre-calculated
        
        ps = 1;
        for  pcons = 1:3
            ps = ps .* (t(pcons).^2);
        end
        
        %Nominator
        nom = 0;
        for signom = 1:3
            top = (D(signom) .* v(signom) .* ps) ./ (t(signom).^2);
            nom = nom + top ; 
        end
        
        %Denominator
        denum = 0;
        for sigdenum  = 1:3
            bot = (D(sigdenum) .* ps) ./ (t(sigdenum).^2);
            denum = denum + bot ;
        end
        v_m  = nom / denum   ;   %Mean Poisson's Ratio

        %Now, calculate the Cubic Spring coefficient
        cc = 0;
        for sigcc  = 1:3
            inda = D(sigcc) .* ps ./(t(sigcc).^2);
            cc = cc + inda;
        end
        
        const = (81 .* pi) .* (-2109.* v_m.^2 + 3210 .* v_m + 5679)./...
            (625 .* (R.^2));
        
        k_cubic = const * cc ./ ps %% CUBIC SPRING COEFFICIENT


        %% Pressure for transient flow
        

        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %               DETERMINATION OF CAPACITANCE
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %       Pressure as a function of average deflection
               
       
        Capacitance  =  [];        % Store each iterated Capacitance value
        CapacitanceTouchMode = []; % Store Capacitance in Touch Mode
        w_0array =  [];            % Store deflection 
        PTouchmode   =  [];        % Store Pressure in Touch mode
        indice = [];               % Store incide in Touch mode
        
        xtm_Array = [];

        for pressure = 1:length(P)
            q  = @(w_avg) ( (k_lin.* w_avg) + (k_cubic.*(w_avg.^3)) ) ./...
                (pi.* (R.^2)); %Pressure formula
            
            funct = @(w_avg) q(w_avg) - P(pressure);
            
            defl_av = fzero(funct, [0 50e-6]); %Determine averg deflection

            w_0 = 3*defl_av; %Maximum deflection not regarding BC's

            w_0array = cat(2, w_0array, w_0);

            constant1 = 1 - sqrt(g./w_0);
            
            if w_0 < g %Determining xtm, x- touchmode coordinate
                xtm = 0;     %This means touch mode is not present
                xtm_Array = cat(2, xtm_Array, xtm)
            else
                xtm = R.*sqrt(constant1); %Coordinate of last touched position
                xtm_Array = cat(2, xtm_Array, xtm)
            end  
            
            
            %% Constants for Capacitance formula
            h = g + ((t(3)+t_4)./e_p);      %For determining capacitance
            
            %   DIAPHRAGM IN CONTACT. w_0 = g
            C_Contact = (pi .* xtm.^2 .*e_0) ./ (h-g);%First term of the eqn

            constant2 = 2 .* pi .* e_0;
            
            %   DIAPHRAGM NOT IN CONTACT
            %   Deflection is determined by formula Thus, Capacitance is
            %   determined by formula 

            C_NoContact = constant2 .* integral(@(r)  r ./ (h - w_0 .*....
                ( (1- (r.^2/R.^2)).^2 )), xtm, R );

            
            C = C_Contact + C_NoContact;
            Capacitance = cat(2,Capacitance, C); %Capacitance array

            % Determine Touchmode Pressure
            
            if g<w_0 
                %Pressure at tM
                PTouchmode = cat(2, PTouchmode, P(pressure)); 
                CapacitanceTouchMode =cat(2,CapacitanceTouchMode,C.*(10.^15));
                indice = cat(2, indice, pressure);
                
            end
            %}

        end

        %F to fF conversion
        Capacitance = Capacitance.* (10.^15);
        
        %

        

        %Array of data on non-touch mode
        if g < w_0array(end)
            
            %Array of data on nonTM
            PnonT = P(1:indice(1) );
            CapnonT = Capacitance(1:indice(1) );
            valr = ones(size(PnonT));
            arr = cat(1, PnonT, valr);
            arr = arr'
            regn = lsqlin(arr, CapnonT, [],[], [PnonT(end),1], CapnonT(end))
            
            fit1 = polyval(regn, PnonT)

            % R^2
            %SStot = sum((CapnonT-mean(CapnonT)).^2);   % Total Sum-Of-Squares
            %SSres = sum((CapnonT-fit1).^2);        %Residual Sum-Of-Squares
            %Rsq1 = 1-SSres/SStot                   % R^2
            %rsqnt = cat(2,rsqnt, Rsq1)
           
            Sensitivitynontm = regn(1) .* 100;
            Sensitivity1 = cat(2, Sensitivity1, Sensitivitynontm)                       
            
            ivntm = regn(2);
            ntzero = cat(2, ntzero, ivntm)

            %Array of data on Touch mode
            P_tM =  PTouchmode;
            Cap_tM = CapacitanceTouchMode;
            valtm = ones(size(PTouchmode));
            arrtm = cat(1, P_tM, valtm);
            arrtm = arrtm'
            regtm = lsqlin(arrtm, Cap_tM, [], [], [P_tM(1),1], Cap_tM(1));
            fit2 = polyval(regtm, P_tM);     %fit for tM
            

            % R^2
            %SStot2 = sum((Cap_tM-mean(Cap_tM)).^2);   % Total Sum-Of-Squares
            %SSres2 = sum((Cap_tM-fit2).^2);        %Residual Sum-Of-Squares
            %Rsq2 = 1-SSres2/SStot2;                   % R^2
            %rsqtm = cat(2, rsqtm, Rsq2)

            Sensitivitytm = regtm(1).*100;
            Sensitivity2 = cat(2, Sensitivity2, Sensitivitytm);
            
            ivtm = regtm(2);
            tzero = cat(2, tzero, ivtm);


            %R2nt = {sprintf('R_1^2 = %f', Rsq1)};
            %R2tm = {sprintf('R_2^2 = %f', Rsq2)};
    
            
            plot(PnonT, fit1, "--", "Color", "black","LineWidth",2);
            %text(P(1),Capacitance(end), R2nt,'right')
            plot(P_tM, fit2, "--","Color","black","LineWidth",2);
            %text(P(70),Capacitance(end), R2tm,"right")
        else
            %Data on NonTouch mode
             PnonT = P;
             CapnonT = Capacitance;
             valr = ones(size(PnonT));
             arr = cat(1, PnonT, valr);
             arr = arr';
             regn = lsqlin(arr, CapnonT, [],[], [PnonT(end),1], CapnonT(end))
            
             fit1 = polyval(regn, PnonT);
             % R^2
             %SStot = sum((CapnonT-mean(CapnonT)).^2);   % Total Sum-Of-Squares
             %SSres = sum((CapnonT-fit1).^2);        %Residual Sum-Of-Squares
             %Rsq1 = 1-SSres/SStot                   % R^2
             
             %rsqnt = cat(2,rsqnt, Rsq1)

             ivntm = regn(2);
             ntzero = cat(2, ntzero, ivntm)

             Sensitivitynontm = regn(1) .* 100;
             Sensitivity1 = cat(2, Sensitivity1, Sensitivitynontm)
             Sensitivity2 = cat(2, Sensitivity2, 0);
             
             %text(P(1),Capacitance(1),R2prt,'right')

             
             plot(PnonT, fit1, "--", "Color", "black","LineWidth",3);
             %R2nt = {sprintf('R_1^2 = %f', Rsq1)};
             %text(P(3),Capacitance(end), R2nt,'right')
            
        end

        %R2nontouch = ntParameters.rsquared
        %R2touch = tParameters.rsquared
         %   if R2touch == NaN
          %      R2touch = 0
           % end
        %}    

        
        %
        %% Plotting Values 
        %
        %cagri400 = um4um1
        %cagrifit = 0.0216.*P + 245.05;
        %cagrifitTM = 0.0548.*P(61:end) + 432 ;

        %titleCell = {sprintf('Yarıçap = %2.1f μm', R*10^6)};
        
        %plot(P, Capacitance, "LineWidth",2, "Color",color{j});
        %plot(P, cagri400, "Color", "red", "LineWidth",2)
        %plot(P, cagrifit, "Color", "red", 'LineStyle','--')
        %plot(P(61:end), cagrifitTM, "Color", "red", 'LineStyle','--')
        
        %plot(P,cagri400)
        %xlabel("Basınç (Pa)")
        %ylabel("Kapasitans (fF)")
        %grid on
        %title(titleCell);
        
        
       % if g < w_0array(end)
        %    xline(PTouchmode(1), "Color", color4 , "LineStyle","--");
         %   yline(CapacitanceTouchMode(1), "Color",color4,"LineStyle","--");
       % end
        %x0=15;
        %y0=5;
        %width=14;
        %height=12;

        %ylim([200 500])
        

        %set(gcf, 'renderer', 'painters',"units","centimeters",'position',[x0,y0,width,height])
        %set(gca,'XTick',0:1000:10000,'YTick',200:50:500,...
         %'fontsize',12, ...
        %'XGrid', 'on', 'YGrid','on','YMinorGrid','off',...
        %'Yminortick','on','Xminortick','on',...        
        %'linewidth',2.0,'GridLineStyle','--',"FontName","Calibri")
        
              
        
        %}
        Sensitivity1
    ntzero
    rsqnt
    
    Sensitivity2
    tzero
    rsqtm
        
        
    end
    Sensitivitynontouch = cat(1, Sensitivitynontouch, Sensitivity1);
    Sensitivitytouch = cat(1, Sensitivitytouch, Sensitivity2);

end
%/
%figure
%contourf(a, t_3.*(10.^6), transpose(Sensitivitynontouch),30)
%xlabel('Radius um');
%ylabel('Thickness um');
%title('Contour Plot of Sensitivity for nontouch mode (fF/100Pa)');
%colorbar;
%figure
%contourf(a, t_3.*(10.^6), transpose(Sensitivitytouch),30)
%xlabel('Radius um');
%ylabel('Thickness um');
%title('Contour Plot of Sensitivity for touch mode (fF/100Pa)');
%/%
%colorbar; % Add a colorbar to indicate values

%hold off