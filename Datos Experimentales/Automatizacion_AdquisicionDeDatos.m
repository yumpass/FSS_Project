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
number = 1;
for k = 1:epocas
    for i = ti:pasos_t:tf

        fprintf('T = %05.2f°C\n', i);
        writeline(mega, sprintf('%.2f', i));               % Enviar setpoint de temperatura al arduino
        



        pause(t_est);                                        % Espera tiempo de estabilización (obtenido experimentalmente)
        set=false;
        while set == false
            
            pause(0.5)                             % leer incompleto
            line = strtrim(readline(mega));
            if startsWith(line,"T=")
                temp_now = str2double(extractAfter(line,"T="));
            else
                temp_now = NaN;
            end
            fprintf('Recibido: "%s"  ->  T = %g; SP = %05.2f°C\n', line, temp_now, i);

            
            if abs(temp_now - i) < umbral
                set = true;
            end
        end

        % ------------------------Tomar foto thorlabs---------------------
        cam.Acquisition.Freeze(uc480.Defines.DeviceParameter.DontWait);

        [~, tmp] = cam.Memory.CopyToArray(MemId);
        Data = reshape(uint8(tmp), [Bits/8, Width, Height]);
        Data = Data(1:3, 1:Width, 1:Height);
        Data = permute(Data, [3,2,1]);

        % -----------------------------------------------------------------
        speckle(:,:,:,number,k) = Data;           % Guarda la foto de la camara
        flush(mega)
        readline(mega);
        timeout_s = 3;
        t0 = tic;
        temp = NaN;
    
        while toc(t0) < timeout_s
            line = strtrim(readline(mega));   % ejemplo: "T=25.34" o "SP=30.00"
            
            if startsWith(line, "T=")
                temp = str2double(extractAfter(line, "T="));
                if ~isnan(temp)
                    T(number,k) = temp; % Guarda temperatura real
                end
            end
            % si llega SP= o basura, la ignoramos y seguimos leyendo
        end


        % ---------------------- Guardando dataset en .tiff ---------------
        if T(number,k) < 100
            file_name_1 = sprintf('C:/Users/VA/Desktop/Isaac Huertas/FSS_Project/Datos Experimentales/Datasets/4/FSS2_exp_%06.1fnm_0%05.2fC_%04.0f.tiff', Lamda, T(number,k), number);
        else
            file_name_1 = sprintf('C:/Users/VA/Desktop/Isaac Huertas/FSS_Project/Datos Experimentales/Datasets/4/FSS2_exp_%06.1fnm_%05.2fC_%04.0f.tiff', Lamda, T(number,k), number);
        end
        imwrite(speckle(:,:,:, number, k), file_name_1);
        % -----------------------------------------------------------------
        number = number + 1;

    end
end

write(mega,"25","string");                  % Enviar setpoint de temperatura al arduino
cam.Exit;