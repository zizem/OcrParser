package org.example.Screen;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.File;


public class ScreenSelector extends JFrame {
    private Point startPoint;
    private Rectangle selection;

    public ScreenSelector() {

        setUndecorated(true);
        setBackground(new Color(0, 0, 0, 50)); // Полупрозрачный черный
        setExtendedState(JFrame.MAXIMIZED_BOTH);
        setAlwaysOnTop(true);
        setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));

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

    private void captureScreen(Rectangle area) {
        try {
            if (area.width <= 0 || area.height <= 0) return;


            setVisible(false);


            Thread.sleep(150);

            Robot robot = new Robot();


            BufferedImage screenshot = robot.createScreenCapture(area);
            System.out.println("Скриншот сделан! Размер: " + screenshot.getWidth() + "x" + screenshot.getHeight());

            File file = new File("/home/zizen/Cource/JavaFinal/src/main/java/NewSeria.png");
            ImageIO.write(screenshot, "png", file);


            screenshot.flush();


        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}