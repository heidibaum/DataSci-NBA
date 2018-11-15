function playerList = loadPlayerList(n)
%LOADPLAYERLIST reads the fixed player list in players.txt and returns a
%cell array of the words


%% Read the fixed player list
fid = fopen('players.txt');

playerList = cell(n, 4);
for i = 1:n
    % Read line
    line = fgets(fid);
    % segment player name and team and add to list
    playerList(i,:) = strsplit(line,',');

end
fclose(fid);

end
