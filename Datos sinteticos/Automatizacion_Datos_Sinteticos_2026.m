%inicia el contador de tiempo para saber el calculo cuanto dura
tic;
clear all
clc;

% === Conexión a COMSOL ===
import com.comsol.model.util.*; % utilidades LiveLink
% Carga modelo ya configurado en COMSOL
model = mphload("C:\Users\Lab Optica\OneDrive - INSTITUTO TECNOLOGICO METROPOLITANO - ITM\Semestre 2025-2\Fundamentos de Fotónica FUNFOT04-1\Datos sinteticos\FMF_Temperature_Variation_V2_2026.mph");

%% --- Parámetros del problema ---
Lambda =  632.8e-9;           % [m]
r_core = 62.5e-6/2;                   % [m] radio del nucleo
r_cladd = 120e-6/2;                    % [m] radio revestimiento
Z  = 10e-3;                  % 0.3 mm en metros (longitud de perturbación)
T0 = 24;                      % [°C] Temperatura inicial
temps = 0:0.1:176;            % [°C] rango de temperatura con step de 0.1
CTO  = 10e-6;                %coeficiente termo-optico del nucleo %11.9e-6;              
CTO_cladd = 10.5e-6;          %coeficiente termo-optico del revestimiento silica glass(SiO2)
L = 0.2;                      % [m] 10cm de largo de fibra
R = 100e9;%[5e-3, 10e-3, 15e-3, 20e-3];                    % [m] radio de curvatura efectivo (eje x radial saliente)
NA0 = 0.33;


%% --- Malla de muestreo para el speckle (plano de salida) ---
% Rejilla cartesiana en la sección (para mphinterp)
Deltax=0.1*1e-6;
x0=-r_core:Deltax:r_core; % ± radio del nucleo
y0=-r_core:Deltax:r_core; % ± radio del nucleo

length_x0=length(x0);
length_y0=length(y0);

[X,Y]=meshgrid(x0,y0);

XY=[X(:),Y(:)]'; % 2 x (Nx*Ny)

% Tag del estudio/dataset (ajusta si cambiaste etiquetas)
stdTag  = 'std1';              %'std1';
dsetTag = 'dset1';             % por defecto en estudios nuevos suele ser dset1

%% --- Funciones auxiliares ---
get_beta = @(m) mphglobal(model, 'ewfd.beta', 'solnum', m, 'dataset', dsetTag); % β_m
% Campos modales (componentes complex) en la rejilla:
get_Ex = @(m) mphinterp(model,'ewfd.Ex','coord',XY, ...
                        'dataset',dsetTag,'solnum',m,'complexout','on');
get_Ey = @(m) mphinterp(model,'ewfd.Ey','coord',XY, ...
                        'dataset',dsetTag,'solnum',m,'complexout','on');
get_Ez = @(m) mphinterp(model,'ewfd.Ez','coord',XY, ...
                        'dataset',dsetTag,'solnum',m,'complexout','on');
model.param.set('Core_d', sprintf('%.12g[m]', r_core*2)); 
model.param.set('Cladd_d', sprintf('%.12g[m]', r_cladd*2));
% --- Prealoca resultado para guardar secuencia de specklegramas ---
speck_first = []; speck_last = [];

%constantes de sellmeier para silice fundido desde el handbook
B1=0.6961663;
B2=0.4079426;
B3=0.8974794;
C1=0.0684043^2;
C2=0.1162414^2;
C3=9.896161^2;

%% === Loop en longitud de onda ===
for L_ciclo = 1:numel(Lambda)
    %calcula el índice inicial del nucleo con la formula de sellmeier with Lamda in µm
    n_core0=sqrt( 1 + B1*(1e6*Lambda(L_ciclo))^2/((1e6*Lambda(L_ciclo))^2-C1) + B2*(1e6*Lambda(L_ciclo))^2/((1e6*Lambda(L_ciclo))^2-C2) + B3*(1e6*Lambda(L_ciclo))^2/((1e6*Lambda(L_ciclo))^2-C3) );
    n_clad0=sqrt(n_core0^2-NA0^2);%calcula índice inicial del revestimiento de la fibra con la NA
    model.param.set('n_core', n_core0);
    model.param.set('n_cladd', n_clad0);
    model.param.set('wl', Lambda(L_ciclo));
    %% === Loop en curvatura ===
    for j = 1:numel(R)
        fprintf("Usando archivo de índice: %s\n", model.func('int2').getString('filename'));
        %% === Cuadrícula cartesiana para cubrir el disco de radio b ===
        % Margen pequeño para evitar extrapolación en las fronteras
        pad = 0.5e-6; 
        xv = linspace(-(r_cladd+pad), r_cladd+pad, 801);   % denso para suavidad modal
        yv = linspace(-(r_cladd+pad), r_cladd+pad, 801);
        [Xs,Ys] = meshgrid(xv, yv);
        
        rho   = hypot(Xs,Ys);
        theta = atan2(Ys,Xs);
        %% === Variacion de Temperatura ===
        for k = 1:numel(temps)
            Tval = T0 + temps(k);
            n_core = (n_core0+CTO*(Tval-T0));
            n_clad = (n_clad0+CTO_cladd*(Tval-T0));
            model.param.set('n_core', n_core);
            model.param.set('n_cladd', n_clad);
            %% === Perfil base n0(r): step-index ===
            %Step-index simple:
            n0 = n_clad*ones(size(Xs));
            n0(rho <= r_core) = n_core;
            %% === Corrección geométrica por curvatura (mapeo conforme) ===
            nG = n0.* (1 + Xs./R(j));
            nExport = nG;
        
            %% === Enmascara fuera del disco (solo cladding <= b) ===
            mask = (rho <= r_cladd);
            nExport(~mask) = NaN;   % Fuera del dominio no lo usaremos
            
            %% === Empaquetar tabla (x, y, n) solo en el disco ===
            xList = Xs(mask);
            yList = Ys(mask);
            nList = nExport(mask);
            
            nTable = [xList(:), yList(:), nList(:)];
        
            %% === Enviar n(x,y) a COMSOL como Interpolation 2D (n_interp) ===
            tmpfile = fullfile(tempdir, sprintf('nxyn_fiber_curved_linear.txt', R(j)*1e3));
            fid = fopen(tmpfile,'w');
            fprintf(fid, '%% x[m]\ty[m]\tn\n');
            fclose(fid);
            writematrix(nTable, tmpfile, ...
            'FileType', 'text', ...
            'WriteMode', 'append', ...
            'Delimiter', '\t');
            
            model.func('int2').set('filename', tmpfile); 
            model.func('int2').refresh;
            %% === Calculo cantidad de modos ===
            newNA=sqrt(n_core^2-n_clad^2);%calcula la nueva apertura numerica la cual cambia por los efectos termicos
            V = (2*pi*r_core/Lambda(L_ciclo))*newNA;
            M_est = round(V^2/2);
            % Cantidad de modos
            nModesToUse = min(25, max(8, M_est));% razonable para pocos modo
            model.study('std1').feature('mode').set('neigs',nModesToUse);

            model.study(stdTag).run;
            
            %% === Calculo de intensidad del speckle ===
            betas = zeros(1,nModesToUse);% Suma coherente E(x,y,z=Z)
            Ex_sum = zeros(length_y0,length_x0);   
            Ey_sum = zeros(length_y0,length_x0);  
            Ez_sum = zeros(length_y0,length_x0);  
            for m = 1:nModesToUse
                betas(m) = get_beta(m);
                Ex = get_Ex(m); Ex = Ex(:);   
                Ey = get_Ey(m); Ey = Ey(:); 
                Ez = get_Ez(m); Ez = Ez(:); 
                Emag = sqrt(abs(Ex).^2 + abs(Ey).^2 + abs(Ez).^2);     % magnitud local del campo modal
                Max_Emag = max(Emag);
                Ex = Ex/Max_Emag;
                Ey = Ey/Max_Emag;
                Ez = Ez/Max_Emag;
                Ex_rs = reshape(Ex, length_y0, length_x0);
                Ey_rs = reshape(Ey, length_y0, length_x0);
                Ez_rs = reshape(Ez, length_y0, length_x0);
                if mod(temps(k),5) == 0
                    magE_md = abs(Ex_rs).^2 + abs(Ey_rs).^2 + abs(Ez_rs).^2;
                    outdir = 'C:\Users\Lab Optica\Desktop\Isaac Huertas\FSS_Proyect\Datos sinteticos\Datasets\3\Modos';
                    fname = sprintf('M%d-%02d_WL%04dnm_T%05.1fC.jpg', m, nModesToUse, Lambda(L_ciclo)*1e9, Tval); %sprintf('R%06.2fmm_T%05.1fC_M%d-%02d.jpg', R_curv*1e3, Tval, m, nModesToUse);  % p.ej. 17_25.0.jpg
                    imwrite(magE_md, fullfile(outdir, fname));%Iu8, fullfile(outdir, fname));
                    
                end
                Ex_sum = Ex_sum + Ex_rs .* exp(1i*betas(m)*Z);%C(m) .* Ex_rs .* exp(1i*betas(m)*Z);
                Ey_sum = Ey_sum + Ey_rs .* exp(1i*betas(m)*Z);%C(m) .* Ey_rs .* exp(1i*betas(m)*Z);
                Ez_sum = Ez_sum + Ez_rs .* exp(1i*betas(m)*Z);%C(m) .* Ez_rs .* exp(1i*betas(m)*Z);
            end
        
            magE = abs(Ex_sum).^2 + abs(Ey_sum).^2 + abs(Ez_sum).^2;
        
            I = abs(magE);      % intensidad (speckle)
            Phase = angle(magE);% fase del campo
            Iimg = reshape(I, length_y0, length_x0);
        
            % Guarda un par de ejemplos
            if k==1,   speck_first = Iimg; end
            if k==numel(temps), speck_last = Iimg; end
            
            % === Guardar imagen grayscale del speckle por iteración (sin IPT) ===
            outdir = 'C:\Users\Lab Optica\Desktop\Isaac Huertas\FSS_Proyect\Datos sinteticos\Datasets\3';
            if k == 1 && ~exist(outdir,'dir'), mkdir(outdir); end
            
            % (Opcional) comprimir rango dinámico con raíz para resaltar granos débiles:
            I = sqrt(max(Iimg,0));     % si no quieres compresión, usa: I = max(Iimg,0);
            
            % Normaliza a [0,1] manualmente y convierte a uint8
            Imin = min(I(:));
            Imax = max(I(:));
            if ~isfinite(Imin) || ~isfinite(Imax) || Imax <= Imin
                Iu8 = zeros(size(I), 'uint8');         % evita división por cero / NaN
            else
                Iu8 = uint8(255 * (I - Imin) / (Imax - Imin));
            end
            
            % Nombre y escritura del JPG (grayscale por ser matriz 2D)
            fname = sprintf('WL%04dnm_T%05.1fC_M%02d.jpg', Lambda(L_ciclo)*1e9, Tval, nModesToUse);%_M%02d.jpg',Lambda(L_ciclo)*1e9, R*1e3, Tval, nModesToUse);
            imwrite(Iu8, fullfile(outdir, fname));%Iu8, fullfile(outdir, fname));
        
            % (Opcional) aquí puedes calcular correlaciones con un patrón referencia, etc.
        end
    end    
end