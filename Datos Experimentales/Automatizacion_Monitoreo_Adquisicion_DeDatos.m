clear all; close all; clc;

%% ------------- Parámetros ------------------------------------------
ti = 25;                   % Temperatura inicial [°C]
tf = 150;                  % Temperatura final [°C]
pasos_t = 0.175;           % Paso de temperatura [°C]
Lamda = 633;               % Longitud de onda [nm]
Z = 12;                    % Longitud de zona sensora [mm]
t_est = 20;                % Tiempo de estabilización [seg]

epocas = 2;

umbral = 0.5;              % Umbral de temperatura aceptable [°C]

mega = serialport("COM5",9600);             % serial arduino
configureTerminator(mega,"LF");
flush(mega)
speckle = zeros(1024,1280,3,floor((tf-ti)*pasos_t)+1, epocas,'uint8');
T = zeros(floor((tf-ti)/pasos_t)+1, epocas);

%% ------------------------------- INCIAR CÁMARA THORLABS---------------------------------------
NET.addAssembly('C:\Program Files\Thorlabs\Scientific Imaging\DCx Camera Support\Develop\DotNet\uc480DotNet.dll');

cam = uc480.Camera;
cam.Init(0);
cam.Display.Mode.Set(uc480.Defines.DisplayMode.DiB);
cam.PixelFormat.Set(uc480.Defines.ColorMode.RGBA8Packed);
cam.Trigger.Set(uc480.Defines.TriggerMode.Software);
[~, MemId] = cam.Memory.Allocate(true);
[~, Width, Height, Bits, ~] = cam.Memory.Inquire(MemId);
cam.Acquisition.Freeze(uc480.Defines.DeviceParameter.Wait);
[~, tmp] = cam.Memory.CopyToArray(MemId);
pause(1)

%% ---------------------------------- Adquisicion -----------------------------------------
%% ------------- Parámetros Actualizados ------------------------------------------
% ... (tus parámetros iniciales ti, tf, pasos_t se mantienen igual) ...

% Definir el camino: de ti a tf, y de tf de vuelta a ti
vector_t = [ti:pasos_t:tf, (tf-pasos_t):-pasos_t:ti];
num_pasos_total = length(vector_t);

% Ajustamos el tamaño de las matrices para el ciclo completo (Subida + Bajada)
speckle = zeros(1024,1280,3, num_pasos_total, epocas, 'uint8');
T_real = zeros(num_pasos_total, epocas); 

% ... (Código de inicialización de cámara Thorlabs igual) ...


%% ---------------------------------- Adquisición -----------------------------------------
figure('Name', 'Monitoreo de Histéresis en Tiempo Real');
hAni = animatedline('Color', 'r', 'LineWidth', 1.2, 'DisplayName', 'Temp Real'); 
hSP = animatedline('Color', 'b', 'LineStyle', '--', 'DisplayName', 'Setpoint');
legend; grid on; xlabel('Muestras'); ylabel('°C');

contador_muestras = 0;
historial_log = [];

for k = 1:epocas
    number = 1; % Reiniciamos el contador de paso por época
    for i = vector_t
        
        % --- Determinar si es Subida o Bajada para el nombre ---
        % Si el índice es menor o igual a la mitad del vector, es subida
        if number <= floor(num_pasos_total/2) + 1
            fase = 'Subida';
        else
            fase = 'Bajada';
        end
        
        fprintf('Época %d | %s | Target: %.2f°C\n', k, fase, i);
        writeline(mega, sprintf('%.2f', i)); 
        
        pause(t_est); 
        
        set_ok = false;
        while ~set_ok
            line = strtrim(readline(mega));
            if startsWith(line,"T=")
                temp_now = str2double(extractAfter(line,"T="));
                if ~isnan(temp_now)
                    % Gráfica dinámica
                    contador_muestras = contador_muestras + 1;
                    addpoints(hAni, contador_muestras, temp_now);
                    addpoints(hSP, contador_muestras, i);
                    drawnow limitrate
                    
                    % Log de datos para exportar
                    historial_log = [historial_log; contador_muestras, i, temp_now, k];
                    
                    % Verificación de umbral
                    if abs(temp_now - i) < umbral
                        set_ok = true; 
                    end
                end
            end
        end
        
        % ------------------------ Captura de Imagen ---------------------
        cam.Acquisition.Freeze(uc480.Defines.DeviceParameter.Wait);
        [~, tmp] = cam.Memory.CopyToArray(MemId);
        Data = reshape(uint8(tmp), [Bits/8, Width, Height]);
        Data = Data(1:3, 1:Width, 1:Height);
        Data = permute(Data, [3,2,1]);
        
        % Guardar en matriz y archivos
        T_real(number, k) = temp_now;
        speckle(:,:,:,number,k) = Data;
        
        % --- Nombre del archivo con Fase (Subida/Bajada) ---
        file_name = sprintf('FSS2_%s_Epo%d_Step%04d_%.2fC.tiff', fase, k, number, temp_now);
        ruta_base = 'C:/Users/VA/Desktop/Isaac Huertas/FSS_Project/Datos Experimentales/Datasets/6/';
        imwrite(Data, fullfile(ruta_base, file_name));
        
        number = number + 1;
        flush(mega);
    end
end

% Guardar resultados finales
save('Resultados_Histeresis.mat', 'historial_log', 'T_real');
writematrix(historial_log, 'Log_Histeresis.csv');


write(mega,"25","string");                  % Enviar setpoint de temperatura al arduino
cam.Exit;