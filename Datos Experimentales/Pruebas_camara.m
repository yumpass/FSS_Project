clear all; close all; clc;

%% --------- Cargar librería Thorlabs ----------
NET.addAssembly( ...
    'C:\Program Files\Thorlabs\Scientific Imaging\DCx Camera Support\Develop\DotNet\uc480DotNet.dll');

%% --------- Inicializar cámara ----------
cam = uc480.Camera;
cam.Init(0);   % 0 = primera cámara detectada

cam.Display.Mode.Set(uc480.Defines.DisplayMode.DiB);
cam.PixelFormat.Set(uc480.Defines.ColorMode.RGBA8Packed);
cam.Trigger.Set(uc480.Defines.TriggerMode.Software);

% Memoria de imagen
[~, MemId] = cam.Memory.Allocate(true);
[~, Width, Height, Bits, ~] = cam.Memory.Inquire(MemId);

%% --------- Ventana de visualización ----------
hFig = figure('Name','Live View - Thorlabs Camera');
hImg = imshow(zeros(Height, Width, 3, 'uint8'));

disp('📷 Cámara activa. Cierre la ventana para salir.');

%% --------- Loop de visualización ----------
while ishandle(hFig)

    cam.Acquisition.Freeze(uc480.Defines.DeviceParameter.DontWait);
    [~, tmp] = cam.Memory.CopyToArray(MemId);

    % Convertir a imagen RGB
    Data = reshape(uint8(tmp), [Bits/8, Width, Height]);
    Data = Data(1:3, 1:Width, 1:Height);
    Data = permute(Data, [3 2 1]);

    set(hImg, 'CData', Data);
    drawnow;
end

%% --------- Cerrar cámara ----------
cam.Exit;
disp('🛑 Cámara cerrada correctamente.');
