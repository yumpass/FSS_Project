%% --- Cargar y Procesar Datos ---
load('Resultados_Histeresis.mat'); % Carga historial_log y T_real
% Estructura de historial_log: [Muestra, SP, TempReal, Epoca]

% Supongamos que ya procesaste tus imágenes y tienes un vector 'medida_optica'
% Si aún no lo tienes, aquí simularemos uno basado en la intensidad media:
% medida_optica = squeeze(mean(mean(mean(speckle,1),2),3)); 

num_pasos = size(T_real, 1);
mitad = floor(num_pasos / 2) + 1;

% Separar temperaturas de la primera época (k=1)
temp_subida = T_real(1:mitad, 1);
temp_bajada = T_real(mitad+1:end, 1);

% --- Graficar Curva de Histéresis ---
figure('Color', 'w', 'Name', 'Análisis de Histéresis Térmica');
hold on;

% Subida
plot(temp_subida, medida_optica(1:mitad), '-or', 'LineWidth', 1.5, 'MarkerFaceColor', 'r', 'DisplayName', 'Subida (Calentamiento)');

% Bajada
plot(temp_bajada, medida_optica(mitad+1:end), '-ob', 'LineWidth', 1.5, 'MarkerFaceColor', 'b', 'DisplayName', 'Bajada (Enfriamiento)');

grid on;
xlabel('Temperatura Real Medida [°C]');
ylabel('Respuesta Óptica (Speckle)');
title('Ciclo de Histéresis del Sensor');
legend('Location', 'best');

% Añadir flechas indicativas de dirección
annotation('arrow', [0.3 0.4], [0.5 0.6], 'Color', 'r'); % Subida
annotation('arrow', [0.6 0.5], [0.4 0.3], 'Color', 'b'); % Bajada

hold off;