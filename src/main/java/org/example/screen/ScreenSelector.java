package org.example.screen;

import org.example.clipBoard.CopyToClipBoard;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;


public class ScreenSelector extends JFrame {
    private Point startPoint;
    private Rectangle selection;
    private Runnable onDone;

    public ScreenSelector() {

        setUndecorated(true);
        setBackground(new Color(0, 0, 0, 50)); // Полупрозрачный черный
        setExtendedState(JFrame.MAXIMIZED_BOTH);
        setAlwaysOnTop(true); //завжди перекриває основний
        setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));

        //тригер на нажаття та відпускання мишки
        addMouseListener(new MouseAdapter() {
            public void mousePressed(MouseEvent e) {
                startPoint = e.getPoint();
                selection = new Rectangle(startPoint);
            }

            public void mouseReleased(MouseEvent e) {
                captureScreen(selection);
                dispose();
            }
        });
        //тригер на рух миші та нормалізація
        addMouseMotionListener(new MouseMotionAdapter() {
            public void mouseDragged(MouseEvent e) {

                int x = Math.min(startPoint.x, e.getX());
                int y = Math.min(startPoint.y, e.getY());
                int width = Math.abs(startPoint.x - e.getX());
                int height = Math.abs(startPoint.y - e.getY());

                selection.setBounds(x, y, width, height);
                repaint();
            }
        });
    }

    //
    @Override
    public void paint(Graphics g) {
        super.paint(g);
        if (selection != null) {
            Graphics2D g2d = (Graphics2D) g;
            g2d.setColor(Color.RED);
            g2d.draw(selection);
            g2d.setColor(new Color(255, 255, 255, 50));
            g2d.fill(selection);
        }
    }
    //метод для захвату екрана
    private void captureScreen(Rectangle area) {
        try {
            if (area.width <= 0 || area.height <= 0) return;


            setVisible(false);


            Thread.sleep(150); // затримку так як метод працює асинхронно

            Robot robot = new Robot();


            BufferedImage screenshot = robot.createScreenCapture(area);

            System.out.println("Скриншот сделан! Размер: " + screenshot.getWidth() + "x" + screenshot.getHeight());

//            File file = new File("screenshot.png");
//            ImageIO.write(screenshot, "png", file);

            CopyToClipBoard.copyToClipBoard(screenshot);


            screenshot.flush(); //очистка щоб не забивати памьять
            if (onDone != null) onDone.run();


        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public void setOnDone(Runnable onDone) {
        this.onDone = onDone;
    } //маркер для того щоб показати що скріншот готовий
}
